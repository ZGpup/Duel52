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

use std::io::{self, BufRead, Write};

use duel52_engine::agents::Agent;
use duel52_engine::display::{describe_action, power_reference, render};
use duel52_engine::{
    stats, GameConfig, GameState, Outcome, Player, RandomAgent, TwoPower, Variant, VERSION,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("help");

    let result = match command {
        "play" => cmd_play(&args[1..]),
        "stats" => cmd_stats(&args[1..]),
        "demo" => cmd_demo(&args[1..]),
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
  duel52 powers                   print the card-power reference
  duel52 config  <file>           validate a config file and print what it resolves to
  duel52 version

OPTIONS
  --variant <base|split|mirrored> which configuration (default: split — the project default)
  --two-power <bottom|discard>    the 2's power; `bottom` is the house rule (default)
  --seed <n>                      game seed; the same seed always deals the same game
  --config <file>                 load a config file (overrides --variant/--two-power)
  --stalemate <plies>             quiet plies before a draw is declared (default 20)

  play only:
  --as <p0|p1>                    which side you take (default: p0, who moves first)
  --opponent <random|human>       `human` is hotseat: both sides played at the keyboard
  --reveal                        DEBUG: show all hidden information (both hands, base
                                  cards, pile order). Do not use while genuinely playing.

  stats only:
  --games <n>                     games per configuration (default: 2000)
  --all                           run all three variants, and both settings of the 2
  --markdown                      emit a Markdown table row, for pasting into FINDINGS.md

EXAMPLES
  duel52 play --seed 1                     play the default variant as first player
  duel52 play --variant base --as p1       play the rules-as-written game, second
  duel52 stats --all --games 5000
"
    )
}

// ============================================================== argument parsing ==

/// Options gathered from the command line.
struct Options {
    config: GameConfig,
    seed: u64,
    human: Player,
    hotseat: bool,
    reveal: bool,
    games: usize,
    all: bool,
    markdown: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            config: GameConfig::default(),
            seed: 1,
            human: Player::P0,
            hotseat: false,
            reveal: false,
            games: 2000,
            all: false,
            markdown: false,
        }
    }
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
            "--games" => opts.games = next_number(args, &mut i, "--games")?,
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
                opts.hotseat = match v.to_ascii_lowercase().as_str() {
                    "random" | "bot" => false,
                    "human" | "hotseat" => true,
                    other => {
                        return Err(format!("--opponent expects random or human, got `{other}`"))
                    }
                };
            }
            "--reveal" => opts.reveal = true,
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

    let runs = if opts.all {
        stats::phase1_sweep(opts.seed, opts.games)
    } else {
        vec![stats::run_random_games(opts.config, opts.seed, opts.games)]
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

/// Interactive play.
fn cmd_play(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let mut state = GameState::new(opts.config, opts.seed);
    let mut bot = RandomAgent::derived(opts.seed ^ 0xBEEF, 99);

    println!("duel52 engine {VERSION}");
    println!("config: {}", state.config.summary());
    println!("seed:   {} (replay this exact game with --seed {})", state.seed, state.seed);
    if opts.hotseat {
        println!("mode:   hotseat — both sides played at this keyboard");
    } else {
        println!(
            "mode:   you are {}, the opponent plays uniformly at random",
            opts.human
        );
    }
    if opts.reveal {
        println!("REVEAL MODE IS ON — hidden information is being printed.");
    }
    println!("Type `help` at any prompt. Type `q` to quit.\n");

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    while !state.outcome.is_over() {
        let acting = state.acting_player();
        let human_turn = opts.hotseat || acting == opts.human;

        if !human_turn {
            let legal = state.legal_actions();
            let action = bot.choose(&state, &legal);
            println!(
                "-- {acting} (bot): {}",
                describe_action(&state, action, Some(acting))
            );
            state.apply_trusted(action);
            continue;
        }

        // The board is always rendered from the acting player's point of view, so hotseat
        // play does not leak either side's hand to the other.
        let observer = if opts.reveal { None } else { Some(acting) };
        println!("{}", render(&state, observer));

        let legal = state.legal_actions();
        for (i, action) in legal.iter().enumerate() {
            println!("  {:>3}. {}", i, describe_action(&state, *action, observer));
        }
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
                println!("{}", play_help());
                continue;
            }
            "powers" => {
                print!("{}", power_reference());
                continue;
            }
            "board" => continue, // the loop re-renders
            "rules" => {
                println!("{}", rules_reminder());
                continue;
            }
            _ => {}
        }

        let Ok(index) = input.parse::<usize>() else {
            println!("!! `{input}` is not a number. Type the number of an action, or `help`.");
            continue;
        };
        let Some(&action) = legal.get(index) else {
            println!("!! {index} is out of range (0..{})", legal.len() - 1);
            continue;
        };

        // `apply` rather than `apply_trusted`: this input came from a human.
        if let Err(e) = state.apply(action) {
            println!("!! {e}");
        }
    }

    println!("{}", render(&state, None));
    println!("=== {} ===", state.outcome);
    match state.outcome {
        Outcome::Win(w) if !opts.hotseat && w == opts.human => println!("You win."),
        Outcome::Win(_) if !opts.hotseat => println!("You lose."),
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
Commands at the prompt:
  <number>   take that action from the numbered list
  help / ?   this message
  powers     the card-power reference
  rules      the handful of rules people get wrong
  board      redraw the board
  q          quit

Reading the board:
  [K]        face-up card, rank K, visible to both players
  (K)        face-down card whose rank YOU know (you played it, or a 4 showed it to you)
  (?)        face-down card whose rank nobody has told you — including your own base cards
  #2         the slot number to use when choosing an attacker or target
  base       still a base card: untouchable until every draw pile is empty
  ex-base    entered play as a base card and was moved by a Queen; you still cannot see it
  dmg1       one damage, on a card whose rank you do not know (max HP would give it away)
  1/3hp      one hit point left of three — so this is a Jack
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
  * Every card has 2 HP. The Jack has 3, face-down included.
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
