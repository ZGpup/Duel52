//! `duel52` — the text CLI.
//!
//! Two jobs:
//!
//! 1. **Let the project owner play the engine and spot-check the rules.** That is Phase 1's
//!    exit criterion in `PLAN.md`: "the owner plays a few games against the CLI and finds no
//!    rules errors." Every prompt names the rule it is applying, so a disagreement is easy
//!    to point at.
//! 2. **Produce the Phase 1 deliverable** — random-vs-random statistics across all three
//!    variants.
//!
//! Run `duel52 help` for usage.

use std::io::{self, BufRead, IsTerminal, Write};

use duel52_engine::agents::Agent;
use duel52_engine::display::{describe_action, describe_move, power_reference, render};
use duel52_engine::{
    ladder, menu, stats, Action, AgentSpec, GameConfig, GameState, Outcome, Player, Rank,
    RandomAgent, TwoPower, Variant, VERSION,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("help");

    let result = match command {
        "play" => cmd_play(&args[1..]),
        "stats" => cmd_stats(&args[1..]),
        "demo" => cmd_demo(&args[1..]),
        "ladder" => cmd_ladder(&args[1..]),
        "match" => cmd_match(&args[1..]),
        "probe" => cmd_probe(&args[1..]),
        "powers" => {
            print!("{}", power_reference());
            Ok(())
        }
        "config" => cmd_config(&args[1..]),
        "version" | "--version" | "-V" => {
            println!("duel52 engine {VERSION}");
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print!("{}", usage());
            Ok(())
        }
        other => Err(format!("unknown command `{other}`\n\n{}", usage())),
    };

    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

fn usage() -> String {
    format!(
        "duel52 engine {VERSION} — a rules-exact Duel 52 engine

USAGE
  duel52 play    [options]        play against the engine in the terminal
  duel52 demo    [options]        watch one random-vs-random game, ply by ply
  duel52 stats   [options]        random-vs-random statistics (Phase 1 deliverable)
  duel52 ladder  [options]        round-robin Elo over the agent ladder (Phase 2)
  duel52 match   [options]        one head-to-head, with behavioural statistics
  duel52 probe   [options]        self-play instrumentation per agent (Phase 2 findings)
  duel52 powers                   print the card-power reference
  duel52 config  <file>           validate a config file and print what it resolves to
  duel52 version

AGENTS
  random            uniform over legal actions
  greedy            one-ply lookahead over a hand-written evaluation
  flatmc[:playouts] random playouts per action, no tree            (default 600)
  pimc[:worldsxdepth]  alpha-beta per sampled world                (default 8x1)
  ismcts[:iters]    information-set MCTS, random rollouts          (default 800)

OPTIONS
  --variant <base|split|mirrored> which configuration (default: split — the project default)
  --two-power <bottom|discard>    the 2's power; `bottom` is the house rule (default)
  --seed <n>                      game seed; the same seed always deals the same game
  --config <file>                 load a config file (overrides --variant/--two-power)
  --stalemate <plies>             quiet plies before a draw is declared (default 20)

  play only:
  --as <p0|p1>                    which side you take (default: p0, who moves first)
  --opponent <agent|human>        an agent name, or `human` for hotseat (default: random)
  --reveal                        DEBUG: show all hidden information (both hands, base
                                  cards, pile order). Do not use while genuinely playing.
  --no-clear                      do not redraw over the screen; keep every prompt in the
                                  scrollback, which is what you want when checking a rule

  stats only:
  --all                           run all three variants, and both settings of the 2

  ladder / match / probe / stats:
  --games <n>                     games per pairing (rounded up to even; default: 2000
                                  for stats, 400 for ladder/match/probe)
  --threads <n>                   worker threads (default: all cores). Results are
                                  identical whatever this is set to.
  --agents <a,b,...>              roster for ladder/probe (default: the frozen ladder)
  --a <agent> --b <agent>         the two sides of a `match`
  --markdown                      emit Markdown, for pasting into FINDINGS.md

EXAMPLES
  duel52 play --seed 1                     play the default variant as first player
  duel52 play --opponent ismcts:2000       play against a stronger search
  duel52 stats --all --games 5000
  duel52 ladder --games 600 --markdown     the Phase 2 Elo table
  duel52 match --a ismcts:800 --b pimc:8x1 --games 400
  duel52 probe --games 400                 hand size, lane and flip statistics per agent
"
    )
}

// ============================================================== argument parsing ==

/// Options gathered from the command line.
struct Options {
    config: GameConfig,
    seed: u64,
    human: Player,
    /// `None` means hotseat — both sides played at the keyboard.
    opponent: Option<AgentSpec>,
    reveal: bool,
    /// Keep the transcript instead of redrawing over it. `play` only.
    no_clear: bool,
    /// `None` means "the command's own default", which differs between `stats` (cheap) and
    /// the Phase 2 commands (expensive).
    games: Option<usize>,
    all: bool,
    markdown: bool,
    threads: usize,
    /// `None` means [`AgentSpec::LADDER`], the frozen benchmark.
    roster: Option<Vec<AgentSpec>>,
    agent_a: Option<AgentSpec>,
    agent_b: Option<AgentSpec>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            config: GameConfig::default(),
            seed: 1,
            human: Player::P0,
            opponent: Some(AgentSpec::Random),
            reveal: false,
            no_clear: false,
            games: None,
            all: false,
            markdown: false,
            threads: default_threads(),
            roster: None,
            agent_a: None,
            agent_b: None,
        }
    }
}

impl Options {
    fn games_or(&self, fallback: usize) -> usize {
        self.games.unwrap_or(fallback)
    }

    fn roster(&self) -> Vec<AgentSpec> {
        self.roster
            .clone()
            .unwrap_or_else(|| AgentSpec::LADDER.to_vec())
    }
}

/// One worker per core. A ladder is minutes of work and the shards are independent, so
/// there is no reason to leave cores idle by default.
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Fetch the value that follows a `--flag`, advancing the cursor.
fn next_value(args: &[String], i: &mut usize, name: &str) -> Result<String, String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| format!("{name} needs a value"))
}

fn next_number<T: std::str::FromStr>(
    args: &[String],
    i: &mut usize,
    name: &str,
) -> Result<T, String> {
    let raw = next_value(args, i, name)?;
    raw.parse()
        .map_err(|_| format!("{name}: `{raw}` is not a number"))
}

/// Parse `--flag value` pairs and boolean flags. Deliberately tiny; the engine has no
/// dependencies, and this is a developer tool.
///
/// Config resolution, in order: `--config` **or** `--variant` picks the base (giving both
/// is an error, since it is never clear which should win), then `--two-power` and
/// `--stalemate` override individual fields.
fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut opts = Options::default();
    let mut from_file: Option<GameConfig> = None;
    let mut variant: Option<Variant> = None;
    let mut two_power: Option<TwoPower> = None;
    let mut stalemate: Option<u32> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--variant" => {
                let v = next_value(args, &mut i, "--variant")?;
                variant = Some(
                    Variant::parse(&v)
                        .ok_or_else(|| format!("unknown variant `{v}` (base|split|mirrored)"))?,
                );
            }
            "--two-power" => {
                let v = next_value(args, &mut i, "--two-power")?;
                two_power = Some(
                    TwoPower::parse(&v)
                        .ok_or_else(|| format!("unknown two-power `{v}` (bottom|discard)"))?,
                );
            }
            "--stalemate" => stalemate = Some(next_number(args, &mut i, "--stalemate")?),
            "--seed" => opts.seed = next_number(args, &mut i, "--seed")?,
            "--games" => opts.games = Some(next_number(args, &mut i, "--games")?),
            "--threads" => {
                let n: usize = next_number(args, &mut i, "--threads")?;
                opts.threads = n.max(1);
            }
            "--agents" => {
                let v = next_value(args, &mut i, "--agents")?;
                let mut roster = Vec::new();
                for name in v.split(',').filter(|s| !s.trim().is_empty()) {
                    roster.push(AgentSpec::parse(name)?);
                }
                if roster.len() < 2 {
                    return Err("--agents needs at least two agents".to_string());
                }
                opts.roster = Some(roster);
            }
            "--a" => opts.agent_a = Some(AgentSpec::parse(&next_value(args, &mut i, "--a")?)?),
            "--b" => opts.agent_b = Some(AgentSpec::parse(&next_value(args, &mut i, "--b")?)?),
            "--config" => {
                let path = next_value(args, &mut i, "--config")?;
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("cannot read `{path}`: {e}"))?;
                from_file = Some(
                    GameConfig::from_config_str(&text).map_err(|e| format!("`{path}`: {e}"))?,
                );
            }
            "--as" => {
                let v = next_value(args, &mut i, "--as")?;
                opts.human = match v.to_ascii_lowercase().as_str() {
                    "p0" | "0" | "first" => Player::P0,
                    "p1" | "1" | "second" => Player::P1,
                    other => return Err(format!("--as expects p0 or p1, got `{other}`")),
                };
            }
            "--opponent" => {
                let v = next_value(args, &mut i, "--opponent")?;
                opts.opponent = match v.to_ascii_lowercase().as_str() {
                    "human" | "hotseat" => None,
                    other => Some(AgentSpec::parse(other)?),
                };
            }
            "--reveal" => opts.reveal = true,
            "--no-clear" => opts.no_clear = true,
            "--all" => opts.all = true,
            "--markdown" | "--md" => opts.markdown = true,
            other => return Err(format!("unknown option `{other}`")),
        }
        i += 1;
    }

    opts.config = match (from_file, variant) {
        (Some(_), Some(_)) => {
            return Err("give either --config or --variant, not both".to_string())
        }
        (Some(cfg), None) => cfg,
        (None, Some(v)) => GameConfig::preset(v),
        (None, None) => opts.config,
    };
    if let Some(t) = two_power {
        opts.config.two_power = t;
    }
    if let Some(s) = stalemate {
        opts.config.stalemate_quiet_plies = s;
    }
    opts.config
        .validate()
        .map_err(|e| format!("bad config: {e}"))?;
    Ok(opts)
}

// ====================================================================== commands ==

fn cmd_config(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("config needs a file path")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read `{path}`: {e}"))?;
    let config = GameConfig::from_config_str(&text).map_err(|e| format!("`{path}`: {e}"))?;
    println!("{path} is valid. It resolves to:\n");
    print!("{}", config.to_config_string());
    println!(
        "\nderived: draw pile {} card(s) per {}",
        config.expected_pile_size(),
        if config.variant.is_split() {
            "player"
        } else {
            "game (shared)"
        }
    );
    Ok(())
}

fn cmd_stats(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let games = opts.games_or(2000);

    let runs = if opts.all {
        stats::phase1_sweep(opts.seed, games)
    } else {
        vec![stats::run_random_games(opts.config, opts.seed, games)]
    };

    if opts.markdown {
        println!("{}", stats::RandomPlayStats::markdown_header());
        for run in &runs {
            println!("{}", run.markdown_row());
        }
    } else {
        for run in &runs {
            print!("{}", run.report());
            println!();
        }
    }
    Ok(())
}

/// The Phase 2 deliverable: a round-robin over the agent ladder, and the Elo table fitted
/// to it.
fn cmd_ladder(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let roster = opts.roster();
    let games = opts.games_or(400);

    if !opts.markdown {
        eprintln!(
            "Round robin: {} agents, {} games per pairing, {} thread(s).",
            roster.len(),
            games + (games % 2),
            opts.threads
        );
    }
    let result = ladder::run_ladder(
        opts.config,
        &roster,
        opts.seed,
        games,
        opts.threads,
        "random",
        !opts.markdown,
    );

    if opts.markdown {
        print!("{}", result.markdown());
    } else {
        print!("{}", result.report());
        println!("Per-pairing detail:\n");
        for m in &result.matches {
            print!("{}", m.report());
            println!();
        }
    }
    Ok(())
}

/// One head-to-head, reported in full. `--a` and `--b` default to the two ends of the
/// ladder, which is the comparison worth running if you did not say.
fn cmd_match(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let a = opts.agent_a.unwrap_or(AgentSpec::Ismcts {
        iterations: duel52_engine::IsmctsAgent::DEFAULT_ITERATIONS,
    });
    let b = opts.agent_b.unwrap_or(AgentSpec::Random);
    let games = opts.games_or(400);

    let result = ladder::run_match(opts.config, a, b, opts.seed, games, opts.threads);
    print!("{}", result.report());
    Ok(())
}

/// Self-play instrumentation, one row per agent.
///
/// `PLAN.md` Phase 2's deliverable is "first real strategic observations", and the way to
/// get them is to watch each rung play *itself* — a mixed pairing measures how an agent
/// copes with a weaker opponent, which is a different and less interesting question than how
/// it plays when the opposition is competent.
fn cmd_probe(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let roster = opts.roster();
    let games = opts.games_or(400);

    let mut rows = Vec::new();
    for spec in &roster {
        if !opts.markdown {
            eprintln!("  {} self-play …", spec.name());
        }
        rows.push(ladder::run_match(
            opts.config,
            *spec,
            *spec,
            opts.seed,
            games,
            opts.threads,
        ));
    }

    println!("Self-play instrumentation — {} games each", games + (games % 2));
    println!("  config: {}\n", opts.config.summary());
    if opts.markdown {
        println!(
            "| agent | P0 score (95% CI) | draws | stalemate | mean plies | hand@unlock | \
             won − lost | flip rate | lane conc | attack conc | passes/game | max lane |"
        );
        println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for m in &rows {
            let b = &m.behaviour[0];
            let (won, lost) = b.hand_at_unlock_by_result();
            println!(
                "| {} | {:.4} ± {:.4} | {:.1}% | {:.1}% | {:.1} | {:.2} | {:+.2} ± {:.2} | {:.3} | {:.3} | {:.3} | {:.2} | {} |",
                m.agents[0].name(),
                m.first_player_score(),
                m.first_player_score_ci95(),
                100.0 * m.draw_rate(),
                100.0 * m.stalemate_rate(),
                m.mean_plies(),
                b.mean_hand_at_unlock(),
                won - lost,
                b.hand_at_unlock_gap_ci95(),
                b.flip_rate(),
                b.mean_lane_concentration(),
                b.mean_attack_concentration(),
                b.passes_per_game(),
                m.max_side_occupancy,
            );
        }
        let rank_header = || {
            print!("| agent |");
            for r in Rank::ALL {
                print!(" {r} |");
            }
            println!();
            print!("|---|");
            for _ in Rank::ALL {
                print!("---:|");
            }
            println!();
        };

        println!("\nMean ply at which each rank is turned face-up\n");
        rank_header();
        for m in &rows {
            print!("| {} |", m.agents[0].name());
            for r in Rank::ALL {
                match m.behaviour[0].mean_flip_ply(r) {
                    Some(ply) => print!(" {ply:.1} |"),
                    None => print!(" — |"),
                }
            }
            println!();
        }

        println!(
            "\nFraction of each rank played from hand that was then turned face-up\n\
             (base-card flips excluded; a 3 that springs its Trap is counted nowhere)\n"
        );
        rank_header();
        for m in &rows {
            print!("| {} |", m.agents[0].name());
            for r in Rank::ALL {
                match m.behaviour[0].flip_rate_for(r) {
                    Some(rate) => print!(" {rate:.2} |"),
                    None => print!(" — |"),
                }
            }
            println!();
        }
    } else {
        for m in &rows {
            print!("{}", m.report());
            let b = &m.behaviour[0];
            print!("  flip ply by rank:");
            for r in Rank::ALL {
                match b.mean_flip_ply(r) {
                    Some(ply) => print!(" {r}={ply:.1}"),
                    None => print!(" {r}=—"),
                }
            }
            println!("\n");
        }
    }
    Ok(())
}

/// Watch one random game. Useful for eyeballing whether the engine's *sequences* look sane,
/// which is a different check from whether any single rule is right.
///
/// Always omniscient: this is a debugging view, and it uses the same RNG streams as
/// `duel52 stats`, so `demo --seed N` replays exactly the game that `stats` counted for
/// seed N.
fn cmd_demo(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let mut state = GameState::new(opts.config, opts.seed);
    let mut p0 = RandomAgent::derived(opts.seed, stats::AGENT_STREAM_P0);
    let mut p1 = RandomAgent::derived(opts.seed, stats::AGENT_STREAM_P1);

    println!("{}", render(&state, None));
    let mut last_ply = state.ply;
    while !state.outcome.is_over() {
        let legal = state.legal_actions();
        let action = match state.to_move {
            Player::P0 => p0.choose(&state, &legal),
            Player::P1 => p1.choose(&state, &legal),
        };
        let who = state.to_move;
        println!(
            "  ply {} {who}: {}",
            state.ply,
            describe_action(&state, action, None)
        );
        state.apply_trusted(action);
        if state.ply != last_ply {
            last_ply = state.ply;
            println!("{}", render(&state, None));
        }
    }
    println!("{}", render(&state, None));
    println!("Result: {}", state.outcome);
    Ok(())
}

/// One line of the move log.
///
/// Rendered once per point of view, at the moment the move is made. It has to be eager:
/// the position the line describes is gone by the time the line is printed, and the slots
/// it names may have shifted. Keeping all three views is what makes the log safe in a
/// hotseat game, where the screen it lands on belongs to whichever player is now at the
/// keyboard — and that player is not entitled to the other's view.
struct Note {
    prefix: String,
    /// Indexed by [`Player::idx`].
    per_player: [String; 2],
    revealed: String,
}

impl Note {
    fn line(&self, observer: Option<Player>) -> String {
        let body = match observer {
            None => &self.revealed,
            Some(p) => &self.per_player[p.idx()],
        };
        format!(" {}{body}", self.prefix)
    }
}

/// The play screen: a banner, the recent moves, the board, and the decision being asked —
/// redrawn in that order after every selection, so what you are choosing between is always
/// directly under the board it applies to.
struct Screen {
    /// Clear the terminal before each redraw. Off when stdout is not a terminal (a pipe
    /// gets a plain transcript instead of escape codes) and off under `--no-clear`.
    clear: bool,
    banner: String,
    log: Vec<Note>,
}

impl Screen {
    /// How many past moves stay on screen above the board. Enough for the opponent's whole
    /// turn plus your own, which is what you need to see to make sense of the position.
    const LOG_LINES: usize = 10;

    /// Record a move. `state` must still be the position the move was chosen in.
    fn note(&mut self, state: &GameState, prefix: String, action: Action) {
        self.log.push(Note {
            prefix,
            per_player: [
                describe_move(state, action, Some(Player::P0)),
                describe_move(state, action, Some(Player::P1)),
            ],
            revealed: describe_move(state, action, None),
        });
        if self.log.len() > Screen::LOG_LINES {
            self.log.drain(..self.log.len() - Screen::LOG_LINES);
        }
    }

    /// Redraw everything, ending with `menu` immediately below the board.
    fn draw(&self, board: &str, menu: &str, observer: Option<Player>) {
        if self.clear {
            // Home the cursor, then erase. Deliberately not `\x1b[3J`, which would throw
            // away the scrollback along with the screen.
            print!("\x1b[H\x1b[2J");
        }
        println!(" {}", self.banner);
        for note in &self.log {
            println!("{}", note.line(observer));
        }
        print!("{board}{menu}");
    }

    /// Show something that is not the board — the help, the power reference — and wait, so
    /// that the next redraw does not wipe it before it has been read.
    fn interlude(&self, text: &str, lines: &mut impl Iterator<Item = io::Result<String>>) {
        if self.clear {
            print!("\x1b[H\x1b[2J");
        }
        print!("{text}");
        if self.clear {
            print!("\n(press Enter to go back to the board) ");
            io::stdout().flush().ok();
            lines.next();
        }
    }
}

/// Interactive play.
fn cmd_play(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let mut state = GameState::new(opts.config, opts.seed);
    let mut bot = opts
        .opponent
        .map(|spec| spec.build(opts.seed ^ 0xBEEF, 99));

    let mode = match opts.opponent {
        None => "hotseat".to_string(),
        Some(spec) => format!("you are {} vs `{}`", opts.human, spec.name()),
    };
    let mut screen = Screen {
        // Clearing only makes sense on a terminal. Piped output — the way the CLI is
        // scripted in tests — stays a readable transcript.
        clear: !opts.no_clear && io::stdout().is_terminal(),
        // The board's own header already carries the config and the engine version, so the
        // banner adds only what the board does not: how to replay this game, who is who,
        // and how to get help.
        banner: format!(
            "seed {} · {mode}{} · `help` for commands, `q` to quit",
            state.seed,
            if opts.reveal { " · REVEALED" } else { "" },
        ),
        log: Vec::new(),
    };

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    while !state.outcome.is_over() {
        let acting = state.acting_player();
        let human_turn = bot.is_none() || acting == opts.human;

        if let (false, Some(bot)) = (human_turn, bot.as_mut()) {
            let legal = state.legal_actions();
            let action = bot.choose(&state, &legal);
            screen.note(&state, format!("-- {acting} (bot): "), action);
            state.apply_trusted(action);
            continue;
        }

        // The board and the menu are always built from the acting player's point of view,
        // so hotseat play does not leak either side's hand to the other.
        let observer = if opts.reveal { None } else { Some(acting) };
        let legal = state.legal_actions();
        let root = menu::build(&state, &legal, observer);
        let board = render(&state, observer);

        // Walk the tree: pick a card, then pick what it does. `path` is where we are in it.
        // The state cannot change while we navigate, so the tree stays valid throughout —
        // and every step redraws the whole screen rather than printing under the last menu.
        let mut path: Vec<usize> = Vec::new();
        let mut complaint = String::new();
        let action = loop {
            let node = root.at(&path);
            let mut menu_text = node.render();
            if !path.is_empty() {
                menu_text.push_str("     0. back\n");
            }
            if !complaint.is_empty() {
                menu_text.push_str(&format!("\n !! {complaint}\n"));
                complaint.clear();
            }
            screen.draw(&board, &menu_text, observer);
            print!("\n{acting}> ");
            io::stdout().flush().ok();

            let Some(line) = lines.next() else {
                println!("\n(end of input)");
                return Ok(());
            };
            let input = line.map_err(|e| format!("cannot read input: {e}"))?;
            let input = input.trim();

            match input {
                "" => continue,
                "q" | "quit" | "exit" => {
                    println!("Quitting. Replay this game with --seed {}", state.seed);
                    return Ok(());
                }
                "help" | "?" => {
                    screen.interlude(&play_help(), &mut lines);
                    continue;
                }
                "powers" => {
                    screen.interlude(&power_reference(), &mut lines);
                    continue;
                }
                "rules" => {
                    screen.interlude(&rules_reminder(), &mut lines);
                    continue;
                }
                "board" => continue, // the loop redraws it
                "b" | "back" | "0" => {
                    if path.pop().is_none() {
                        complaint = "already at the top. `q` quits.".to_string();
                    }
                    continue;
                }
                _ => {}
            }

            let Ok(number) = input.parse::<usize>() else {
                complaint =
                    format!("`{input}` is not a number. Type a number from the list, or `help`.");
                continue;
            };
            match number.checked_sub(1).and_then(|i| node.picks.get(i)) {
                Some(menu::Pick::Take(action)) => break *action,
                Some(menu::Pick::Open(_)) => path.push(number - 1),
                None => complaint = format!("{number} is out of range (1..{})", node.len()),
            }
        };

        // Logged before it is applied, because that is when the description is accurate.
        let prefix = if bot.is_none() {
            format!("-- {acting}: ")
        } else {
            format!("-- {acting} (you): ")
        };
        screen.note(&state, prefix, action);

        // `apply` rather than `apply_trusted`: this input came from a human. The menu only
        // ever offers actions the engine listed, so a rejection here means the menu and the
        // engine disagree — worth hearing about rather than silently ignoring.
        if let Err(e) = state.apply(action) {
            println!("!! {e}");
        }
    }

    println!("{}", render(&state, None));
    println!("=== {} ===", state.outcome);
    match state.outcome {
        Outcome::Win(w) if opts.opponent.is_some() && w == opts.human => println!("You win."),
        Outcome::Win(_) if opts.opponent.is_some() => println!("You lose."),
        _ => {}
    }
    println!(
        "Lanes won: P0 {} · P1 {} (need {})",
        state.lanes_won_by(Player::P0),
        state.lanes_won_by(Player::P1),
        state.config.lanes_to_win
    );
    println!("Replay this game with --seed {}", state.seed);
    Ok(())
}

fn play_help() -> String {
    "\
Choosing a move — two steps, never more:
  1. pick the CARD you want to act with, from your hand or from your side of the board
  2. pick what it does — which lane to play it into, or which card to attack

  A card whose only legal move is forced skips step 2 and takes it, so the first menu
  spells out what that move is on the line you are picking. A flip is always such a
  move, so its line names the power but not the whole power text — `powers` has that.

  The screen is redrawn after every selection: what you are choosing between is always
  directly below the board it applies to, with the last few moves above it. Run with
  --no-clear to keep every prompt in the scrollback instead.

Commands at the prompt:
  <number>   pick that numbered line
  0 / b      back to the card list (from the second menu)
  help / ?   this message
  powers     the card-power reference
  rules      the handful of rules people get wrong
  board      redraw the screen
  q          quit

Reading the board:
  Lanes are numbered 1-3 and slots from #1, the way you would count them. The engine's own
  indices start at 0; nothing you type here uses them.
  [K]        face-up card, rank K, visible to both players
  (K)        face-down card whose rank YOU know (you played it, or a 4 showed it to you)
  (?)        face-down card whose rank nobody has told you — including your own base cards
  #2         the second card in that lane, counting from the left
  base       still a base card: untouchable until every draw pile is empty
  ex-base    entered play as a base card and was moved by a Queen; you still cannot see it
  1/2hp      one hit point left of two. Every FACE-DOWN card is 2 HP whatever it really is,
             so this tells you nothing about its rank — and a face-down Jack dies to two
             hits like anything else. Flip it and its ceiling rises to 3.
  FROZEN     cannot attack and cannot be flipped, by anyone, until its next turn ends
  pair3      member of declared pair #3: attacks only together, and only a Queen or a
             death can break it
  spent      already attacked this turn
"
    .to_string()
}

fn rules_reminder() -> String {
    "\
The five that trip people up (CLAUDE.md):

1. Your own base cards are hidden from YOU too. That is why a 4 can usefully peek at them.
2. A lane is won only when ALL of these hold: the opponent's side of that lane is empty,
   every draw pile is empty, and the opponent's hand is empty. Win two lanes to win. So
   the whole draw phase is positioning — nothing can be won during it.
3. Ten cards were removed at setup without anyone seeing them. You can never fully account
   for the deck.
4. Suits do not exist here. Only rank matters.
5. The default configuration is the split deck: you own one colour, 26 cards, and draw only
   from your own 13-card pile.

Combat quick reference:
  * Every FACE-DOWN card is a blank 2 HP card, whatever its rank — so nothing about a
     face-down card can be learned by attacking it. Face-up, a card has 2 HP, or 3 for the
     Jack. Flipping a damaged Jack raises its ceiling; the damage stays.
  * A pair costs one action, deals 2 damage to ONE target, and both members spend their
     attack. Rank powers still apply: a pair of 9s deals 4 to a Jack, a pair of 10s is
     forced to split 1+1.
  * Attacking a face-up 8 costs you 1 damage — unless you are a 9.
  * A face-up Jack must die before anything else in his lane can be attacked.
  * A 10 hits two cards for 1 each. A face-up 9 refuses the spread; a lone Jack's taunt
     leaves the second half nowhere to go.
  * An Ace gives +1 action and may attack twice on the turn it is flipped.
  * A King refires your other face-up powers in its lane — not Kings, not 8/9/10/J.
"
    .to_string()
}
