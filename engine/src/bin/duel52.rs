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

use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::path::PathBuf;

use duel52_engine::agents::Agent;
use duel52_engine::display::{
    describe_action, describe_move, power_reference, render, render_focus, Focus,
};
use duel52_engine::nn::Evaluator;
use duel52_engine::{
    ladder, menu, record, stats, Action, AgentSpec, GameConfig, GameRecord, GameState, Outcome,
    Player, Rank, RandomAgent, TwoPower, Variant, VERSION,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("help");

    let result = match command {
        "play" => cmd_play(&args[1..]),
        "replay" => cmd_replay(&args[1..]),
        "stats" => cmd_stats(&args[1..]),
        "demo" => cmd_demo(&args[1..]),
        "ladder" => cmd_ladder(&args[1..]),
        "match" => cmd_match(&args[1..]),
        "probe" => cmd_probe(&args[1..]),
        "nn-dump" => cmd_nn_dump(&args[1..]),
        "selfplay" => cmd_selfplay(&args[1..]),
        "shard" => cmd_shard(&args[1..]),
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
  duel52 replay  [options]        walk a recorded game and see what the net thought
  duel52 demo    [options]        watch one random-vs-random game, action by action
  duel52 stats   [options]        random-vs-random statistics (Phase 1 deliverable)
  duel52 ladder  [options]        round-robin Elo over the agent ladder (Phase 2)
  duel52 match   [options]        one head-to-head, with behavioural statistics
  duel52 probe   [options]        self-play instrumentation per agent (Phase 2 findings)
  duel52 nn-dump [options]        encode + forward-pass dump for the Python parity test
  duel52 selfplay [options]       one generation of self-play, written as a .d52sp shard
  duel52 shard   <file>           print a shard's header and replay it as an integrity check
  duel52 powers                   print the card-power reference
  duel52 config  <file>           validate a config file and print what it resolves to
  duel52 version

AGENTS
  random            uniform over legal actions
  greedy            one-ply lookahead over a hand-written evaluation
  flatmc[:playouts] random playouts per action, no tree            (default 600)
  pimc[:worldsxdepth]  alpha-beta per sampled world                (default 8x1)
  ismcts[:iters]    information-set MCTS, random rollouts          (default 800)
  netpolicy:<path>  a .d52nn checkpoint played by argmax, no search
netmcts:<path>[@sims]  net-guided ISMCTS: PUCT over the policy prior,
                  the value head in place of rollouts             (default 128)

OPTIONS
  --variant <base|split|mirrored> which configuration (default: split — the project default)
  --two-power <bottom|discard>    the 2's power; `bottom` is the house rule (default)
  --seed <n>                      game seed; the same seed always deals the same game
  --config <file>                 load a config file (overrides --variant/--two-power)
  --stalemate <plies>             quiet plies before a draw is declared (default 20)
  --stalemate-value <0..0.5>      what an engine-declared stalemate is worth TO A LEARNER,
                                  to both players (default 0.5 — the same as any draw).
                                  Training sets 0.0; see FINDINGS.md F3.6. Scoring and Elo
                                  are unaffected: a draw is still half a point there.
  --encoding-slots <n>            slots per side per lane the NN encoder reserves
                                  (default 16, FINDINGS.md F2.7). Must match the
                                  checkpoint: it is what fixes obs_dim.

  play only:
  --as <p0|p1>                    which side you take (default: p0, who moves first)
  --opponent <agent|human>        an agent name, or `human` for hotseat (default: random)
  --hint [n]                      before each of your decisions, list the n moves the model
                                  you are playing would consider, best first, with the share
                                  of its search each got and what it thinks you are worth
                                  after it (default 3). It reasons from YOUR information
                                  set, so it cannot see anything you cannot, and it runs on
                                  its own RNG: the opponent plays the same game either way.
                                  A hinted game is marked in --record, because `replay`
                                  scoring your moves against the net means something else
                                  when the net's answer was already on the screen.
  --hint-agent <agent>            who does the advising; implies --hint. Default: whoever
                                  you are playing. Needed for a hotseat game or against a
                                  non-net opponent, and how you ask a bigger budget than the
                                  one you are playing against.
  --reveal                        DEBUG: show all hidden information (both hands, base
                                  cards, pile order). Do not use while genuinely playing.
  --no-clear                      do not redraw over the screen; keep every prompt in the
                                  scrollback, which is what you want when checking a rule.
                                  Turns off the live highlight, which is a redraw per
                                  keystroke. `NO_COLOR` turns off the colour alone.
  --record <file>                 append this game to a JSONL record when it finishes.
                                  A game is (config, seed, chosen indices), so the file
                                  replays it exactly — hidden information included — in a
                                  few hundred bytes. Only finished games are written.

  replay only:
  --record <file>                 the JSONL file written by `play --record` (required)
  --game <n>                      which game in it, 1-based. Omit for the index.
  --node <n>                      also print the full board at that decision node
  --checkpoint <file>             score with this .d52nn instead of the one that was
                                  played, which is how an old game becomes a fixed
                                  evaluation set for a new net
  --sims <n>                      search budget for the second opinion (default: the
                                  recorded opponent's; 0 turns the search off)
  --all-nodes                     score the bot's decisions too, not only yours
  --reveal                        describe moves from ground truth rather than from what
                                  the player could see at the time

  stats only:
  --all                           run all three variants, and both settings of the 2

  ladder / match / probe / stats:
  --games <n>                     games per pairing (rounded up to even; default: 2000
                                  for stats, 400 for ladder/match/probe)
  --threads <n>                   worker threads (default: all cores). Results are
                                  identical whatever this is set to.
  --eval-batch <n>                games kept in flight per worker thread, so that their
                                  network evaluations go through the trunk in one batch.
                                  Applies to selfplay, match, ladder and probe. The batch
                                  is taken across games and never inside a search, so
                                  results are identical whatever this is — it only changes
                                  how fast they arrive. Only `netmcts` agents batch; a
                                  match between two different checkpoints splits its games
                                  between two batches. Costs ~400 KB of live search tree
                                  per game in flight per thread. 1 turns it off
                                  (default 1)
  --agents <a,b,...>              roster for ladder/probe (default: the frozen ladder)
  --anchor <agent>                which rung `ladder` pins to 0 Elo, named exactly as it
                                  appears in --agents (default: random). Elo is only
                                  defined up to a constant, so this is what the table is
                                  measured against.
  --a <agent> --b <agent>         the two sides of a `match`
  --markdown                      emit Markdown, for pasting into FINDINGS.md

  nn-dump only:
  --checkpoint <file>             the .d52nn checkpoint to run (required)
  --out <file>                    where to write the dump (required)
  --max-rows <n>                  rows to keep, sampled evenly across the run (default 512)

  selfplay only:
  --checkpoint <file>             the .d52nn checkpoint to play both sides (required)
  --out <file>                    where to write the .d52sp shard (required)
  --games <n>                     self-play games in this generation (default 1000)
  --sims <n>                      PUCT simulations per decision (default 128)
  --full-search-fraction <f>      playout cap randomisation: the share of decisions given
                                  the full --sims and a policy target. The rest get
                                  --cap-sims and a value target only, which costs nothing
                                  extra because one game has one outcome however little
                                  search produced the position (default 1.0, capping off)
  --cap-sims <n>                  simulations for a capped decision. Must be smaller than
                                  --sims, and is ignored when --full-search-fraction is 1
                                  (default 32)
  --c-puct <f>                    PUCT exploration constant (default 1.25)
  --dirichlet-alpha <f>           root noise concentration (default 0.3)
  --dirichlet-weight <f>          root noise share of the prior (default 0.25)
  --temperature <f>               visit-count exponent while sampling (default 1.0)
  --temperature-decisions <n>     decisions sampled before switching to most-visited
                                  (default 24)
  --generation <n>                stamped into the shard header (default 0)
  --quiet                         no per-batch progress lines

EXAMPLES
  duel52 play --seed 1                     play the default variant as first player
  duel52 play --opponent ismcts:2000       play against a stronger search
  duel52 play --encoding-slots 21 --hint \\
    --opponent netmcts:models/duel52-split-gen031.d52nn@256
                                           decide, then read what it would have played
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
    /// How many moves `play --hint` lists before each of your decisions. `None` is off.
    hint: Option<usize>,
    /// Which agent does the advising. `None` means the one you are playing against, which is
    /// what "what does the model I am playing think?" asks for.
    hint_agent: Option<AgentSpec>,
    reveal: bool,
    /// Keep the transcript instead of redrawing over it. `play` only.
    no_clear: bool,
    /// JSONL file to append finished games to (`play`), or to read them from (`replay`).
    /// `--sims`, `--checkpoint` and `--all-nodes` are peeled off by `cmd_replay` itself,
    /// the way `cmd_selfplay` handles its own, because those names already mean something
    /// else under `selfplay` and `nn-dump`.
    record: Option<PathBuf>,
    /// Which game in a record to walk. 1-based, matching the file's line numbers.
    game: Option<usize>,
    /// `None` means "the command's own default", which differs between `stats` (cheap) and
    /// the Phase 2 commands (expensive).
    games: Option<usize>,
    all: bool,
    markdown: bool,
    threads: usize,
    /// Games kept in flight per worker so their network evaluations batch (`PLAN.md` §4.2d).
    /// A pure speed knob: `rule_2_the_ladder_is_eval_batch_independent` pins that the result
    /// does not move. 1 turns it off.
    eval_batch: usize,
    /// `None` means [`AgentSpec::LADDER`], the frozen benchmark.
    roster: Option<Vec<AgentSpec>>,
    /// Which rung `ladder` pins to 0 Elo. `None` means `random`, the original floor.
    ///
    /// Elo is only defined up to an additive constant, so the anchor is a choice about what
    /// the table is measuring rather than a detail. It became worth naming when the
    /// hand-written rungs saturated: a roster with no `random` in it used to anchor on
    /// whichever agent was listed first, which made the scale depend on argument order.
    anchor: Option<String>,
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
            hint: None,
            hint_agent: None,
            reveal: false,
            no_clear: false,
            record: None,
            game: None,
            games: None,
            all: false,
            markdown: false,
            threads: default_threads(),
            eval_batch: 1,
            roster: None,
            anchor: None,
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
    let mut stalemate_value: Option<f32> = None;
    let mut encoding_slots: Option<usize> = None;

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
            "--stalemate-value" => {
                stalemate_value = Some(next_number(args, &mut i, "--stalemate-value")?)
            }
            "--encoding-slots" => {
                encoding_slots = Some(next_number(args, &mut i, "--encoding-slots")?)
            }
            "--seed" => opts.seed = next_number(args, &mut i, "--seed")?,
            "--games" => opts.games = Some(next_number(args, &mut i, "--games")?),
            "--threads" => {
                let n: usize = next_number(args, &mut i, "--threads")?;
                opts.threads = n.max(1);
            }
            "--eval-batch" => {
                let n: usize = next_number(args, &mut i, "--eval-batch")?;
                opts.eval_batch = n.max(1);
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
            "--anchor" => opts.anchor = Some(next_value(args, &mut i, "--anchor")?),
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
            // The count is optional, because `--hint` on its own is the whole of what anyone
            // wants nine times in ten. The number is consumed only if it *is* a number, so
            // `--hint --seed 3` leaves `--seed` alone rather than eating its flag.
            "--hint" => {
                let n = match args.get(i + 1).and_then(|v| v.parse::<usize>().ok()) {
                    Some(n) => {
                        i += 1;
                        n
                    }
                    None => Advisor::DEFAULT_TOP,
                };
                if n == 0 {
                    return Err("--hint needs at least one move to list".to_string());
                }
                opts.hint = Some(n);
            }
            "--hint-agent" => {
                let v = next_value(args, &mut i, "--hint-agent")?;
                opts.hint_agent = Some(AgentSpec::parse(&v)?);
            }
            "--record" => opts.record = Some(next_value(args, &mut i, "--record")?.into()),
            "--game" => opts.game = Some(next_number(args, &mut i, "--game")?),
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
    if let Some(v) = stalemate_value {
        opts.config.stalemate_value = v;
    }
    if let Some(n) = encoding_slots {
        opts.config.encoding_slots = n;
    }
    // Naming an advisor is a request for hints, so `--hint-agent` alone works and nobody has
    // to discover that it silently did nothing without `--hint` next to it.
    if opts.hint_agent.is_some() && opts.hint.is_none() {
        opts.hint = Some(Advisor::DEFAULT_TOP);
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

    // An anchor that is not in the roster would silently fall back to whichever agent was
    // listed first, and the whole table would be measured against something nobody named.
    let anchor = opts.anchor.as_deref().unwrap_or("random");
    if !roster.iter().any(|a| a.name() == anchor) {
        return Err(format!(
            "--anchor {anchor} is not in the roster: {}",
            roster
                .iter()
                .map(|a| a.name())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

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
        opts.eval_batch,
        anchor,
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
    let a = opts.agent_a.clone().unwrap_or(AgentSpec::Ismcts {
        iterations: duel52_engine::IsmctsAgent::DEFAULT_ITERATIONS,
    });
    let b = opts.agent_b.clone().unwrap_or(AgentSpec::Random);
    let games = opts.games_or(400);

    let result = ladder::run_match(opts.config, a, b, opts.seed, games, opts.threads, opts.eval_batch);
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
            spec.clone(),
            spec.clone(),
            opts.seed,
            games,
            opts.threads,
            opts.eval_batch,
        ));
    }

    println!("Self-play instrumentation — {} games each", games + (games % 2));
    println!("  config: {}\n", opts.config.summary());
    if opts.markdown {
        println!(
            "| agent | P0 score (95% CI) | draws | stalemate | mean plies | hand@unlock | \
             won − lost | flip rate | lane conc | attack conc | stuck/game | max lane |"
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
                b.stuck_turns_per_game(),
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
            "\nWhat became of every 3 played from hand — the one rank whose power needs\n\
             darkness. The Trap column is the only one where holding it paid.\n"
        );
        println!(
            "| agent | 3s played/game | flipped by choice | flipped by a 5 | \
             Trap sprang | still face-down at end | traps/game |"
        );
        println!("|---|---:|---:|---:|---:|---:|---:|");
        for m in &rows {
            let b = &m.behaviour[0];
            match b.three_fates() {
                Some((chosen, by_five, trapped, held)) => println!(
                    "| {} | {:.2} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                    m.agents[0].name(),
                    b.threes_played_per_game(),
                    chosen,
                    by_five,
                    trapped,
                    held,
                    b.traps_per_game(),
                ),
                None => println!("| {} | — | — | — | — | — | — |", m.agents[0].name()),
            }
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
            "  turn {} {who}: {}",
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

// ================================================================ Phase 3: the parity dump ==

/// Play seeded random games, encode every decision node, run the Rust forward pass, and
/// write observations, masks, logits and values for `py/tests/test_parity.py` to check.
///
/// **The direction matters.** Rust produces the observations, because it owns the encoder;
/// PyTorch is the reference for the *forward pass*, because it owns the architecture. So
/// this dump is Rust's answer, and the Python test recomputes the same function from the
/// same checkpoint and the same inputs. Neither side re-derives the other's half.
///
/// Rows are sampled evenly across the whole run rather than taken from the opening nodes: a
/// dump of the first `max_rows` decisions would be twenty near-identical fresh deals, which
/// would exercise almost none of the encoder. Two passes make that possible without holding
/// every observation in memory — the engine is deterministic, so the second pass replays the
/// first exactly, which is the same property `PHASE3_STEP1.md` §4 relies on for the replay
/// buffer.
fn cmd_nn_dump(args: &[String]) -> Result<(), String> {
    let mut checkpoint: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut max_rows = 512usize;

    // Only the flags this command uses, so a typo is an error rather than a silent default.
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => checkpoint = Some(next_value(args, &mut i, "--checkpoint")?),
            "--out" => out_path = Some(next_value(args, &mut i, "--out")?),
            "--max-rows" => max_rows = next_number(args, &mut i, "--max-rows")?,
            other => rest.push(other.to_string()),
        }
        i += 1;
    }
    let opts = parse_options(&rest)?;
    let checkpoint = checkpoint.ok_or("nn-dump needs --checkpoint <path>")?;
    let out_path = out_path.ok_or("nn-dump needs --out <path>")?;
    let games = opts.games_or(20);
    if max_rows == 0 {
        return Err("--max-rows must be at least 1".to_string());
    }

    let weights = duel52_engine::nn::Weights::load(std::path::Path::new(&checkpoint), &opts.config)?;
    let arch = weights.arch;
    let evaluator = duel52_engine::nn::MlpEvaluator::new(weights, &opts.config);

    // Pass 1: how many decision nodes are there? Cheap — no encoding, no forward pass.
    let total: usize = (0..games)
        .map(|g| count_decisions(opts.config, opts.seed + g as u64))
        .sum();
    if total == 0 {
        return Err("the games produced no decision nodes".to_string());
    }
    let stride = total.div_ceil(max_rows).max(1);

    // Pass 2: replay the same games and record every `stride`-th decision.
    let obs_dim = duel52_engine::encode::obs_dim(&opts.config);
    let action_dim = duel52_engine::encode::action_dim(&opts.config);
    let mut observations: Vec<f32> = Vec::new();
    let mut masks: Vec<u8> = Vec::new();
    let mut seen = 0usize;
    let mut rows = 0usize;

    let mut obs = vec![0.0f32; obs_dim];
    let mut mask = vec![false; action_dim];
    for g in 0..games {
        let seed = opts.seed + g as u64;
        let mut state = GameState::new(opts.config, seed);
        let mut rng = duel52_engine::Rng::derive(seed, 0xD09E_0000_0000_0001);
        while !state.outcome.is_over() {
            if seen % stride == 0 && rows < max_rows {
                duel52_engine::encode::encode_observation(
                    &state,
                    state.acting_player(),
                    &mut obs,
                );
                duel52_engine::encode::legal_mask(&state, &mut mask);
                observations.extend_from_slice(&obs);
                masks.extend(mask.iter().map(|&m| u8::from(m)));
                rows += 1;
            }
            seen += 1;
            let legal = state.legal_actions();
            state.apply_trusted(*rng.choose(&legal).expect("a running game has actions"));
        }
    }

    // One forward pass over the whole dump, which is also a smoke test of the batch path.
    let mut logits = vec![0.0f32; rows * action_dim];
    let mut values = vec![0.0f32; rows];
    evaluator.eval_batch(&observations, rows, &mut logits, &mut values);

    let mut header = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(header, "rows={rows}");
    let _ = writeln!(header, "obs_dim={obs_dim}");
    let _ = writeln!(header, "action_dim={action_dim}");
    let _ = writeln!(header, "checkpoint={checkpoint}");
    let _ = writeln!(header, "variant={}", opts.config.variant.label());
    let _ = writeln!(header, "encoding_slots={}", opts.config.encoding_slots);
    let _ = writeln!(header, "games={games}");
    let _ = writeln!(header, "first_seed={}", opts.seed);
    let _ = writeln!(header, "stride={stride}");
    let _ = writeln!(header, "decisions={total}");
    let _ = writeln!(header, "width={}", arch.width);
    let _ = writeln!(header, "blocks={}", arch.blocks);
    let _ = writeln!(header, "value_hidden={}", arch.value_hidden);
    let _ = writeln!(
        header,
        "obs_layout_hash={:016x}",
        duel52_engine::encode::obs_layout_hash(&opts.config)
    );
    let _ = writeln!(
        header,
        "action_layout_hash={:016x}",
        duel52_engine::encode::action_layout_hash(&opts.config)
    );
    // `payload_order` plays the part `param_order` plays in a checkpoint: the reader walks
    // it rather than assuming a layout.
    let _ = writeln!(header, "payload_order=obs:f32,mask:u8,logits:f32,values:f32");

    // Same container as a checkpoint (§1.4), different magic — so a dump handed to
    // `Weights::load` by mistake is rejected by magic rather than misread as weights.
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"D52DMP");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in &observations {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend_from_slice(&masks);
    for v in &logits {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in &values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&out_path, &bytes)
        .map_err(|e| format!("cannot write `{out_path}`: {e}"))?;

    println!(
        "wrote {out_path} — {rows} rows of {} decisions across {games} games (stride {stride}), \
         {:.1} MB",
        total,
        bytes.len() as f64 / 1e6
    );
    println!(
        "  widest lane side the encoder saw: {} (bound {})",
        duel52_engine::encode::observed_max_slots(),
        opts.config.encoding_slots
    );
    Ok(())
}

/// Decision nodes in one seeded random game. The same stream `cmd_nn_dump`'s second pass
/// uses, so the two passes agree exactly.
fn count_decisions(config: GameConfig, seed: u64) -> usize {
    let mut state = GameState::new(config, seed);
    let mut rng = duel52_engine::Rng::derive(seed, 0xD09E_0000_0000_0001);
    let mut n = 0;
    while !state.outcome.is_over() {
        n += 1;
        let legal = state.legal_actions();
        state.apply_trusted(*rng.choose(&legal).expect("a running game has actions"));
    }
    n
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

/// What one read from the prompt produced.
enum Key {
    /// The player typed something that is not yet an answer. Redraw and keep waiting — this
    /// is what puts the live highlight on the board as a number is being typed.
    Edit,
    /// An arrow: walk the cursor one row down the list, or one row up it.
    Step { down: bool },
    /// Left arrow — back up one question, the arrow-key half of `0` / `b`.
    Ascend,
    /// Enter, or the right arrow. The buffer holds the whole line, and an empty buffer means
    /// "whatever the cursor is on" — which is what makes the arrows a complete way to play.
    Submit,
    /// No more input: end of a piped script, or Ctrl-D at an empty prompt.
    Eof,
}

/// The prompt's keyboard, in the two shapes it comes in.
///
/// `Keys` is the interactive one: it takes the terminal out of line mode so that every
/// keystroke arrives immediately, which is what lets the board light up the card a number
/// points at *before* Enter commits to it. Everything else — a pipe, a test script,
/// `--no-clear` — gets `Lines`, which reads whole lines exactly as this CLI always has.
///
/// The two are the same interface on purpose: there is one prompt loop, and the only thing
/// that changes between them is how often it is asked to redraw.
enum Keyboard {
    /// The `RawMode` is never read. It is held because dropping it is what puts the
    /// terminal back the way it was found.
    Keys(#[allow(dead_code)] RawMode),
    Lines(io::Lines<io::StdinLock<'static>>),
}

impl Keyboard {
    /// Interactive if the terminal can support it, line-based otherwise.
    ///
    /// `redraws` is [`Screen::clear`]: without it every keystroke would print a fresh copy
    /// of the whole screen underneath the last one, so the live highlight is exactly as
    /// available as redrawing is.
    fn open(redraws: bool) -> Keyboard {
        match redraws.then(RawMode::enter).flatten() {
            Some(raw) => Keyboard::Keys(raw),
            None => Keyboard::Lines(io::stdin().lock().lines()),
        }
    }

    fn interactive(&self) -> bool {
        matches!(self, Keyboard::Keys(_))
    }

    /// Read until there is something for the caller to do. `buf` is the line so far.
    fn read(&mut self, buf: &mut String) -> io::Result<Key> {
        match self {
            Keyboard::Keys(_) => read_key(buf),
            Keyboard::Lines(lines) => match lines.next() {
                None => Ok(Key::Eof),
                Some(line) => {
                    *buf = line?;
                    Ok(Key::Submit)
                }
            },
        }
    }

    /// Wait for Enter and throw the line away — what an interlude needs.
    fn wait(&mut self) {
        let mut scratch = String::new();
        while !matches!(self.read(&mut scratch), Ok(Key::Submit) | Ok(Key::Eof)) {}
    }
}

/// One keystroke, applied to `buf`.
fn read_key(buf: &mut String) -> io::Result<Key> {
    let Some(byte) = read_byte()? else {
        return Ok(Key::Eof);
    };
    match byte {
        b'\r' | b'\n' => Ok(Key::Submit),
        // Backspace arrives as DEL on every terminal worth naming, and as BS on the rest.
        0x7f | 0x08 => {
            buf.pop();
            Ok(Key::Edit)
        }
        // Ctrl-C. `-isig` means the terminal no longer turns this into a signal, which is
        // deliberate: a signal would kill the process with the terminal still in raw mode.
        // Quitting through the same door as `q` is what puts it back.
        0x03 => {
            *buf = "q".to_string();
            Ok(Key::Submit)
        }
        // Ctrl-D ends the game only at an empty prompt, as a shell would.
        0x04 => match buf.is_empty() {
            true => Ok(Key::Eof),
            false => Ok(Key::Edit),
        },
        // Ctrl-U clears the line.
        0x15 => {
            buf.clear();
            Ok(Key::Edit)
        }
        // An escape sequence — an arrow or a function key. A bare Esc is not followed by
        // `[` or `O`, and then the byte after it is treated normally.
        0x1b => match read_byte()? {
            Some(b'[') | Some(b'O') => Ok(match escape_final()? {
                Some(b'A') => Key::Step { down: false },
                Some(b'B') => Key::Step { down: true },
                // Right selects, left goes back — the same two things Enter and `0` do, so
                // a whole decision can be made from the arrow keys alone.
                Some(b'C') => Key::Submit,
                Some(b'D') => Key::Ascend,
                _ => Key::Edit,
            }),
            Some(other) => Ok(push_key(buf, other)),
            None => Ok(Key::Edit),
        },
        other => Ok(push_key(buf, other)),
    }
}

/// Take a printable character. Anything else is ignored rather than echoed, so a stray
/// control code cannot make the prompt unreadable.
fn push_key(buf: &mut String, byte: u8) -> Key {
    // Long enough for `powers`, and short enough that leaning on a key does nothing
    // interesting.
    if (byte.is_ascii_graphic() || byte == b' ') && buf.chars().count() < 16 {
        buf.push(byte as char);
    }
    Key::Edit
}

/// Read an escape sequence to its end and hand back the byte that identifies it.
///
/// `Esc [` and `Esc O` introduce a sequence of parameter bytes (digits, `;`) terminated by
/// one in `@`..`~` — so an arrow is `Esc [ A` but the same arrow with a modifier held is
/// `Esc [ 1 ; 5 A`, and Delete is `Esc [ 3 ~`. Reading to the terminator is what stops the
/// tail of a longer sequence landing in the prompt as text. `Esc O A` terminates on its
/// first byte, so both forms go through here.
///
/// Capped rather than looping forever, because the bytes are coming from a terminal and a
/// sequence this long is line noise, not a keystroke.
fn escape_final() -> io::Result<Option<u8>> {
    for _ in 0..16 {
        match read_byte()? {
            None => return Ok(None),
            Some(byte) if (0x40..=0x7e).contains(&byte) => return Ok(Some(byte)),
            Some(_) => {}
        }
    }
    Ok(None)
}

/// One byte from the terminal. `None` is end of input.
///
/// Reading one byte at a time is safe because `Stdin`'s buffer is shared and outlives each
/// lock, so a keystroke that arrived alongside others is not lost between calls.
fn read_byte() -> io::Result<Option<u8>> {
    let mut byte = [0u8; 1];
    match io::stdin().read(&mut byte)? {
        0 => Ok(None),
        _ => Ok(Some(byte[0])),
    }
}

/// The terminal in character-at-a-time mode, restored on the way out.
///
/// Done by shelling out to `stty`, because the engine crate has **no dependencies** and this
/// is not the thing to break that for (see the workspace `Cargo.toml`). The cost is one
/// process per decision, against a human deciding what to do, which is nothing.
///
/// Held for the length of one decision rather than the whole game, so that a thinking bot
/// or a crash between turns never leaves the terminal in this state.
struct RawMode {
    /// The settings as they were, in `stty -g` form.
    saved: String,
}

impl RawMode {
    fn enter() -> Option<RawMode> {
        if !io::stdin().is_terminal() {
            return None;
        }
        let saved = stty_saved()?;
        // `-icanon` delivers keystrokes as they are typed, `-echo` lets the redraw print
        // them, and `-isig` hands us Ctrl-C rather than letting it kill us mid-raw-mode.
        // `min 1 time 0` keeps the read blocking, so waiting costs nothing.
        if !stty(&["-icanon", "-echo", "-isig", "min", "1", "time", "0"]) {
            return None;
        }
        Some(RawMode { saved })
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        stty(&[self.saved.as_str()]);
    }
}

/// The current terminal settings, in the opaque form `stty` takes back.
fn stty_saved() -> Option<String> {
    // `output()` would otherwise give `stty` a null stdin, and `stty` reads the terminal it
    // is asked about from there.
    let out = std::process::Command::new("stty")
        .arg("-g")
        .stdin(std::process::Stdio::inherit())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|saved| !saved.is_empty())
}

fn stty(args: &[&str]) -> bool {
    std::process::Command::new("stty")
        .args(args)
        .stdin(std::process::Stdio::inherit())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// The play screen: a banner, the recent moves, the board, and the decision being asked —
/// redrawn in that order after every selection, so what you are choosing between is always
/// directly under the board it applies to.
struct Screen {
    /// Clear the terminal before each redraw. Off when stdout is not a terminal (a pipe
    /// gets a plain transcript instead of escape codes) and off under `--no-clear`.
    clear: bool,
    /// Paint the row being typed, and the cards it names, in red. Same conditions as
    /// `clear`, plus the `NO_COLOR` convention.
    color: bool,
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

    /// Redraw everything, ending with `menu` immediately below the board and the prompt —
    /// carrying whatever has been typed so far — under that.
    ///
    /// One write, and no `\x1b[2J`. In interactive mode this runs on **every keystroke**,
    /// and erasing the screen before redrawing it is what makes that flicker: each line
    /// clears its own tail with `\x1b[K` instead, and `\x1b[J` takes whatever the last
    /// frame left below. The cursor ends up after the typed text, where a cursor belongs.
    fn draw(&self, board: &str, menu: &str, observer: Option<Player>, prompt: &str) {
        let mut frame = format!(" {}\n", self.banner);
        for note in &self.log {
            frame.push_str(&format!("{}\n", note.line(observer)));
        }
        frame.push_str(board);
        frame.push_str(menu);
        frame.push_str(&format!("\n{prompt}"));
        if self.clear {
            // Home the cursor rather than erasing to it. Deliberately not `\x1b[3J`, which
            // would throw away the scrollback along with the screen.
            frame = format!("\x1b[H{}\x1b[K\x1b[J", frame.replace('\n', "\x1b[K\n"));
        }
        print!("{frame}");
        io::stdout().flush().ok();
    }

    /// Show something that is not the board — the help, the power reference — and wait, so
    /// that the next redraw does not wipe it before it has been read.
    fn interlude(&self, text: &str, keys: &mut Keyboard) {
        if self.clear {
            print!("\x1b[H\x1b[2J");
        }
        print!("{text}");
        if self.clear {
            print!("\n(press Enter to go back to the board) ");
            io::stdout().flush().ok();
            keys.wait();
        }
    }
}

// ================================================================== the advisor ==

/// The second opinion `play --hint` puts on screen before you choose.
///
/// The same machinery `replay` scores a *finished* game with, moved to the front of the
/// decision — so that the comparison between what you would play and what the net would play
/// can be made while it still costs something, rather than afterwards when the answer is
/// already known.
///
/// **It cannot cheat, and that is structural rather than a promise.** [`NetMctsAgent::search`]
/// determinizes from the acting player's information set on every simulation, exactly as it
/// does when it is playing the other side, so the worlds it reasons over are worlds you
/// yourself cannot rule out — your own face-down base card included. The policy-only path
/// goes through [`duel52_engine::encode_observation`], which
/// `phase3_observation_is_a_function_of_the_information_set` pins to the same standard. A
/// hint is a stronger player's opinion about your position, never a look at the deal.
///
/// It also cannot change the game. It runs on its own RNG stream, so the opponent's choices
/// are bit-identical whether hints are on or off, and the seed still replays the deal.
struct Advisor {
    /// The agent's name, so a hint on screen says what produced it and a line in
    /// `FINDINGS.md` can name the advisor that was on.
    name: String,
    /// How many moves to list.
    top: usize,
    /// `None` for a policy-only advisor, which has no search to run.
    searcher: Option<duel52_engine::NetMctsAgent>,
    /// Set only when there is no searcher: the policy is then the whole of the opinion.
    evaluator: Option<std::sync::Arc<duel52_engine::nn::MlpEvaluator>>,
}

/// One row of a hint: a move, and why the advisor rates it where it does.
enum Opinion {
    /// A search ran. `share` is this move's slice of the simulations — the rule `choose`
    /// picks by — and `win` is what the search backed up behind it, `None` if it never went
    /// there at all.
    Searched { share: f32, win: Option<f32> },
    /// No search: the policy prior is the whole opinion.
    Prior(f32),
}

/// What the advisor made of one position, ready to draw.
struct Hint {
    agent: String,
    /// The advisor's estimate that **you** win from here, `0.0..=1.0`.
    win: f32,
    /// At most `top` moves, best first.
    moves: Vec<(String, Opinion)>,
}

impl Advisor {
    /// Three: enough to see the runner-up and the one after it, which is where a disagreement
    /// usually is, and short enough to sit under the menu without pushing the board off an
    /// 80×24 terminal.
    const DEFAULT_TOP: usize = 3;

    /// Build the advisor, loading its checkpoint now rather than on the first decision.
    ///
    /// A bad path has to be an error before a card is dealt, for the same reason
    /// `--record` is checked there: discovering it ten moves in costs the game.
    fn new(spec: &AgentSpec, top: usize, config: &GameConfig, seed: u64) -> Result<Advisor, String> {
        let load = |path: &str| {
            duel52_engine::nn::evaluator_for(std::path::Path::new(path), config)
                .map_err(|e| format!("--hint cannot load {path}: {e}"))
        };
        let (searcher, evaluator) = match spec {
            AgentSpec::NetMcts { checkpoint, sims } => {
                load(checkpoint)?;
                // A stream of its own — `0xADD1CE`, and not the opponent's `0xBEEF`. Sharing
                // one would make the bot's play depend on whether you asked for advice,
                // which would put two different games behind the same seed.
                let searcher = duel52_engine::NetMctsAgent::derived(
                    checkpoint.clone(),
                    seed ^ 0xADD1CE,
                    11,
                    *sims,
                );
                (Some(searcher), None)
            }
            AgentSpec::NetPolicy { checkpoint } => (None, Some(load(checkpoint)?)),
            other => {
                return Err(format!(
                    "--hint needs a net to advise with, and `{}` is not one. Either play \
                     against a checkpoint, or name the advisor yourself:\n  \
                     --hint-agent netmcts:models/duel52-split-gen031.d52nn@256",
                    other.name()
                ))
            }
        };
        Ok(Advisor {
            name: spec.name(),
            top,
            searcher,
            evaluator,
        })
    }

    /// Whether a decision costs a search, and so whether it is worth saying so on screen.
    fn searches(&self) -> bool {
        self.searcher.is_some()
    }

    /// What the advisor would play here, best first. `legal` must be `state.legal_actions()`
    /// for `state`, and `observer` is who the moves are described to.
    fn advise(&mut self, state: &GameState, legal: &[Action], observer: Option<Player>) -> Hint {
        let (win, mut ranked) = match self.searcher.as_mut() {
            Some(searcher) => {
                let result = searcher.search(state, legal);
                let total = result.visits.iter().sum::<u32>().max(1) as f32;
                let mut order: Vec<usize> = (0..legal.len()).collect();
                // Most-visited first, which is the robust-child rule `NetMctsAgent::choose`
                // picks by, so the top row of a hint is always the move it would have made.
                // The backed-up value breaks ties; `sort_by` is stable, so the tail of
                // never-visited moves keeps the order the menu offers them in.
                order.sort_by(|&a, &b| {
                    result.visits[b].cmp(&result.visits[a]).then_with(|| {
                        let (x, y) = (result.values[b], result.values[a]);
                        x.unwrap_or(0.0).total_cmp(&y.unwrap_or(0.0))
                    })
                });
                let rows = order
                    .into_iter()
                    .map(|i| {
                        (
                            i,
                            Opinion::Searched {
                                share: result.visits[i] as f32 / total,
                                win: result.values[i],
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                (result.root_value, rows)
            }
            None => {
                let evaluator = self.evaluator.as_ref().expect("one path or the other");
                let mut obs = vec![0.0f32; duel52_engine::obs_dim(&state.config)];
                let mut logits = vec![0.0f32; duel52_engine::action_dim(&state.config)];
                let mut values = [0.0f32; 1];
                duel52_engine::encode_observation(state, state.acting_player(), &mut obs);
                evaluator.eval_batch(&obs, 1, &mut logits, &mut values);
                let priors = masked_softmax(&logits, legal, state);
                let mut order: Vec<usize> = (0..legal.len()).collect();
                order.sort_by(|&a, &b| priors[b].total_cmp(&priors[a]));
                let rows = order
                    .into_iter()
                    .map(|i| (i, Opinion::Prior(priors[i])))
                    .collect::<Vec<_>>();
                // The value head is a tanh in `-1..=1`; everything on screen is a win
                // probability, so put it on the engine's `0..=1` outcome scale.
                (0.5 * (values[0] + 1.0), rows)
            }
        };
        ranked.truncate(self.top);
        Hint {
            agent: self.name.clone(),
            win,
            moves: ranked
                .into_iter()
                .map(|(i, opinion)| (describe_move(state, legal[i], observer), opinion))
                .collect(),
        }
    }
}

impl Hint {
    /// Column width for the move text. Wide enough for the longest thing `describe_move`
    /// writes — an attack naming both cards — and narrow enough that the whole block fits
    /// the 79 columns the rest of the CLI is laid out to.
    const MOVE_WIDTH: usize = 44;

    /// The block that goes between the menu and the prompt.
    fn render(&self) -> String {
        let searched = matches!(self.moves.first(), Some((_, Opinion::Searched { .. })));
        let mut out = format!(
            "\n {} · it puts you at {:.0}% from here{}\n",
            self.agent,
            self.win * 100.0,
            if searched { "" } else { " · no search" },
        );
        // Trimmed, because the policy-only variant leaves the last column empty and a screen
        // that is redrawn on every keystroke should not carry a tail of spaces into the
        // scrollback under --no-clear.
        let row = |a: String, b: String, c: String, d: String| {
            format!(" {a:>2}  {b:<width$}  {c:>6}  {d:>6}", width = Hint::MOVE_WIDTH)
                .trim_end()
                .to_string()
        };
        out.push_str(&row(
            "#".to_string(),
            "the net would consider".to_string(),
            if searched { "sims" } else { "prior" }.to_string(),
            if searched { "after" } else { "" }.to_string(),
        ));
        out.push('\n');
        for (i, (text, opinion)) in self.moves.iter().enumerate() {
            let (left, right) = match opinion {
                // A move the search never took has no value behind it, and `—` says that
                // rather than implying it scored badly.
                Opinion::Searched { share, win } => (
                    format!("{:.0}%", share * 100.0),
                    match win {
                        Some(w) => format!("{:.0}%", w * 100.0),
                        None => "—".to_string(),
                    },
                ),
                Opinion::Prior(p) => (format!("{p:.3}"), String::new()),
            };
            out.push_str(&row(
                (i + 1).to_string(),
                clip(text, Hint::MOVE_WIDTH),
                left,
                right,
            ));
            out.push('\n');
        }
        out
    }
}

/// Interactive play.
fn cmd_play(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    // Before a card is dealt, not after the game is won. A record path that cannot be
    // written is otherwise discovered an hour later, with the game as the price — which is
    // the one outcome this whole feature exists to prevent.
    if let Some(path) = &opts.record {
        record::prepare(path)?;
    }
    // Built before the deal too, and for the same reason: a checkpoint that will not load is
    // a command-line mistake, not something to discover on move ten.
    let mut advisor = match opts.hint {
        None => None,
        Some(top) => {
            let spec = opts
                .hint_agent
                .as_ref()
                .or(opts.opponent.as_ref())
                .ok_or(
                    "--hint has nobody to ask: this is a hotseat game, so there is no \
                     opponent to borrow. Name one with --hint-agent <agent>.",
                )?;
            Some(Advisor::new(spec, top, &opts.config, opts.seed)?)
        }
    };

    let mut state = GameState::new(opts.config, opts.seed);
    let mut bot = opts
        .opponent
        .as_ref()
        .map(|spec| spec.build(opts.seed ^ 0xBEEF, 99));

    let mode = match &opts.opponent {
        None => "hotseat".to_string(),
        Some(spec) => format!("you are {} vs `{}`", opts.human, spec.name()),
    };
    // Named on the frame the player waits in front of, so the wait says who it is on.
    let bot_name = opts
        .opponent
        .as_ref()
        .map(|spec| spec.name())
        .unwrap_or_default();
    let redraws = !opts.no_clear && io::stdout().is_terminal();
    // Why the live highlight is off, if it is. Checked in the order the conditions are
    // imposed, and the last of them by *doing the thing* — entering character-at-a-time
    // mode and leaving again — because a terminal that will not switch is the one case
    // that cannot be predicted from a flag. A player whose board is not lighting up is
    // owed the reason on screen rather than left to wonder whether they built the change.
    let plain: Option<&str> = if opts.no_clear {
        Some("--no-clear")
    } else if !io::stdout().is_terminal() {
        Some("output is not a terminal")
    } else if !io::stdin().is_terminal() {
        Some("input is not a terminal")
    } else if RawMode::enter().is_none() {
        Some("stty will not switch this terminal")
    } else if std::env::var_os("NO_COLOR").is_some() {
        Some("NO_COLOR is set")
    } else {
        None
    };
    let mut screen = Screen {
        // Clearing only makes sense on a terminal. Piped output — the way the CLI is
        // scripted in tests — stays a readable transcript.
        clear: redraws,
        // https://no-color.org — set to anything, and we emit no escape colours.
        color: plain.is_none(),
        // The board's own header already carries the config and the engine version, so the
        // banner adds only what the board does not: how to replay this game, who is who,
        // and how to get help.
        banner: format!(
            "seed {} · {mode}{}{}{} · `help` for commands, `q` to quit",
            state.seed,
            if opts.reveal { " · REVEALED" } else { "" },
            // On screen for the same reason it goes into the record: a game you were being
            // advised through is a different piece of evidence from one you were not.
            if advisor.is_some() { " · HINTED" } else { "" },
            match plain {
                Some(why) => format!(" · no live highlight ({why})"),
                None => String::new(),
            },
        ),
        log: Vec::new(),
    };

    // The whole recording, as it accumulates: one index into `legal_actions()` per decision,
    // both sides. `PLAN.md` §4.0 — the engine is deterministic, so this plus the config and
    // the seed replays the game exactly, hidden information included.
    let mut moves: Vec<usize> = Vec::new();

    while !state.outcome.is_over() {
        let acting = state.acting_player();
        let human_turn = bot.is_none() || acting == opts.human;

        if let (false, Some(bot)) = (human_turn, bot.as_mut()) {
            let legal = state.legal_actions();
            // Your move is on the board *before* the bot starts thinking about it. The frame
            // the human decision loop drew is the position they chose in; without this one,
            // the action they just committed to is not drawn until the bot has answered,
            // which at 4096 simulations is seconds spent looking at a stale board. Same
            // guard as the advisor's thinking frame: under `--no-clear` a superseded frame
            // is a whole board of noise in the transcript rather than a redraw.
            if screen.clear {
                let observer = if opts.reveal { None } else { Some(opts.human) };
                screen.draw(
                    &render(&state, observer),
                    &format!("\n {bot_name} is thinking…\n"),
                    observer,
                    "",
                );
            }
            let action = bot.choose(&state, &legal);
            record_choice(&mut moves, &legal, action);
            screen.note(&state, format!("-- {acting} (bot): "), action);
            state.apply_trusted(action);
            continue;
        }

        // The board and the menu are always built from the acting player's point of view,
        // so hotseat play does not leak either side's hand to the other.
        let observer = if opts.reveal { None } else { Some(acting) };
        let legal = state.legal_actions();
        let root = menu::build(&state, &legal, observer);

        // Ask the advisor before the keyboard opens, not after: a 4096-simulation search is
        // seconds of thinking, and the terminal should not sit in raw mode through it.
        let hint = match advisor.as_mut() {
            // A forced move needs no second opinion. Skipping it also keeps a long chain of
            // sub-decisions — `NEXT` after `NEXT` — from costing a search each to be told
            // the one thing that can happen.
            Some(_) if legal.len() < 2 => None,
            Some(advisor) => {
                // Only when there is a screen to redraw over and something to wait for. A
                // policy-only advisor answers instantly, and under --no-clear a frame that
                // is superseded a moment later is a whole board of noise in the transcript.
                if screen.clear && advisor.searches() {
                    screen.draw(
                        &render(&state, observer),
                        &format!("\n {} is thinking…\n", advisor.name),
                        observer,
                        "",
                    );
                }
                Some(advisor.advise(&state, &legal, observer))
            }
            None => None,
        };

        // Character-at-a-time for the length of this decision only, so that a thinking bot
        // between turns is never waiting with the terminal in raw mode.
        let mut keys = Keyboard::open(screen.clear);

        // Walk the tree: pick a card, then pick what it does. `path` is where we are in it.
        // The state cannot change while we navigate, so the tree stays valid throughout —
        // and every step redraws the whole screen rather than printing under the last menu.
        let mut path: Vec<usize> = Vec::new();
        let mut complaint = String::new();
        let mut typed = String::new();
        // Where the arrow keys have walked to. `None` means "not placed yet", which is what
        // every new question is set back to, and the top of the loop then parks it on the
        // first row that can be picked — so a list always opens with a row marked and Enter
        // always has something to take.
        let mut cursor: Option<usize> = None;
        let action = loop {
            let node = root.at(&path);

            // Only where there is a mark to see. Line mode — a pipe, `--no-clear` — draws no
            // `*` and no red, so a cursor there would be an invisible default that an empty
            // line silently acted on; there, a blank line goes on meaning nothing.
            if cursor.is_none() && keys.interactive() {
                cursor = node.entry(!path.is_empty());
            }

            // The row the prompt is pointing at: the number typed so far, or — with nothing
            // typed — wherever the arrows are sitting. Recomputed on every keystroke, which
            // is the whole point: three identical `(? ²♥)` in an enemy lane are told apart
            // by watching which one turns red, before Enter commits to it.
            let selected = keys
                .interactive()
                .then(|| match typed.trim().is_empty() {
                    true => cursor,
                    false => typed.trim().parse::<usize>().ok(),
                })
                .flatten();
            // The board can only light up on a terminal with colour. The `*` beside the row
            // does not need one, which is what keeps the arrows usable without it.
            let focus = match selected.filter(|_| screen.color) {
                Some(number) => node.focus(&state, number),
                None => Focus::none(),
            };

            let board = render_focus(&state, observer, &focus);
            // The menu stays directly under the board it applies to, and the hint goes below
            // it, so the block that changes as you walk the tree is the one next to the
            // board and the block that does not is the one next to the prompt.
            let mut menu_text = node.render_with(!path.is_empty(), selected, screen.color);
            if let Some(hint) = &hint {
                menu_text.push_str(&hint.render());
            }
            if !complaint.is_empty() {
                menu_text.push_str(&format!("\n !! {complaint}\n"));
            }
            screen.draw(&board, &menu_text, observer, &format!("{acting}> {typed}"));

            let key = match keys.read(&mut typed) {
                Err(e) => return Err(format!("cannot read input: {e}")),
                Ok(key) => key,
            };
            // Up and down move the cursor and left is `back`, so the arrows take the line
            // over — but neither clears the complaint, which belongs to the last thing that
            // was *answered* and has to survive until something answers it.
            match key {
                Key::Eof => {
                    println!("\n(end of input)");
                    print_unrecorded(&opts, state.seed);
                    return Ok(());
                }
                // Still typing. Round the loop to redraw with the new highlight.
                Key::Edit => continue,
                // The walk carries on from whatever is marked on screen — a number already
                // typed as readily as a row already walked onto — so the first arrow after
                // typing moves the mark rather than teleporting it back to the top.
                Key::Step { down } => {
                    typed.clear();
                    cursor = node.step(selected, !path.is_empty(), down);
                    continue;
                }
                // At the top there is nowhere to go back to. Silently, unlike `0` — an arrow
                // held down a moment too long is not a mistake worth a message.
                Key::Ascend => {
                    typed.clear();
                    if path.pop().is_some() {
                        cursor = None;
                    }
                    continue;
                }
                Key::Submit => {}
            }
            let input = std::mem::take(&mut typed);
            let input = input.trim();
            // The complaint has now been read — it survived every keystroke of the line
            // that follows it, and goes when that line is answered.
            complaint.clear();

            match input {
                // Nothing typed: the cursor answers for it, below.
                "" => {}
                "q" | "quit" | "exit" => {
                    println!("\nQuitting. Replay this game with --seed {}", state.seed);
                    print_unrecorded(&opts, state.seed);
                    return Ok(());
                }
                "help" | "?" => {
                    screen.interlude(&play_help(), &mut keys);
                    continue;
                }
                "powers" => {
                    screen.interlude(&power_reference(), &mut keys);
                    continue;
                }
                "rules" => {
                    screen.interlude(&rules_reminder(), &mut keys);
                    continue;
                }
                "board" => continue, // the loop redraws it
                "b" | "back" => {
                    if path.pop().is_none() {
                        complaint = "already at the top. `q` quits.".to_string();
                    }
                    cursor = None;
                    continue;
                }
                _ => {}
            }

            // Enter — or the right arrow — on an empty prompt takes whatever the cursor is
            // on, which is what makes the arrow keys a complete way to play. With no cursor
            // there is nothing to take, and an empty line goes back to the board.
            let number = match (input.is_empty(), cursor) {
                (true, None) => continue,
                (true, Some(number)) => number,
                (false, _) => match input.parse::<usize>() {
                    Ok(number) => number,
                    Err(_) => {
                        complaint = format!(
                            "`{input}` is not a number. Type a number from the list, or `help`."
                        );
                        continue;
                    }
                },
            };
            // `#0` is BACK, whether it was typed or the cursor walked onto it.
            if number == 0 {
                if path.pop().is_none() {
                    complaint = "already at the top. `q` quits.".to_string();
                }
                cursor = None;
                continue;
            }
            match node.picks.get(number - 1) {
                Some(menu::Pick::Take(action)) => break *action,
                Some(menu::Pick::Open(_)) => {
                    path.push(number - 1);
                    // A new question, so nothing is walked onto in it yet.
                    cursor = None;
                }
                // A row marked `—` keeps its number precisely so the others keep theirs, so
                // this is a normal thing to type, not a mistake worth scolding.
                Some(menu::Pick::Unavailable) => {
                    complaint = format!(
                        "nothing available under {} right now.",
                        node.rows[number - 1].name
                    )
                }
                None => complaint = format!("{number} is out of range (1..{})", node.len()),
            }
        };
        // Back to line mode before anything is applied: the rest of the turn belongs to the
        // engine, and a panic there should not land on a terminal that cannot echo.
        drop(keys);

        // Logged before it is applied, because that is when the description is accurate.
        let prefix = if bot.is_none() {
            format!("-- {acting}: ")
        } else {
            format!("-- {acting} (you): ")
        };
        screen.note(&state, prefix, action);

        // Recorded before it is applied, and only if the engine accepts it below — an
        // action the engine rejected was not part of the game and must not be replayed as
        // though it were.
        let chosen = legal.iter().position(|a| *a == action);

        // `apply` rather than `apply_trusted`: this input came from a human. The menu only
        // ever offers actions the engine listed, so a rejection here means the menu and the
        // engine disagree — worth hearing about rather than silently ignoring.
        if let Err(e) = state.apply(action) {
            println!("!! {e}");
        } else if let Some(index) = chosen {
            moves.push(index);
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

    if let Some(path) = &opts.record {
        let record = GameRecord::new(
            opts.config,
            opts.seed,
            opts.human,
            opts.opponent.as_ref().map(|spec| spec.name()),
            moves,
            state.outcome,
        )
        .with_hint(advisor.as_ref().map(|a| a.name.clone()));
        // A finished game is never thrown away because of a filesystem problem. If the
        // append fails anyway — the disk filled, the directory went away mid-game — the
        // line goes to the terminal, where it can be pasted into the file by hand. It is
        // one line, and it is the only copy of a game that took an hour to play.
        if let Err(e) = record.append_to(path) {
            println!("\n!! {e}");
            println!("!! The game is NOT lost. Append this line to the file yourself:\n");
            println!("{}", record.to_json_line().trim_end());
            return Err(e);
        }
        println!(
            "Recorded game {} of {} — {} nodes, {}.",
            record_count(path),
            path.display(),
            record.moves.len(),
            record.human_result
        );
    }
    Ok(())
}

/// Say that a game was abandoned rather than lost, so nobody goes looking for it in the file.
///
/// Only complete games are recorded. A half-played game cannot be checked against an
/// outcome, and the check is the thing that makes a record trustworthy years later — so the
/// invariant is "every line in the file is a whole game that still reproduces", and the
/// price is that quitting loses the game. The seed is printed either way, which is enough to
/// deal the identical position again.
fn print_unrecorded(opts: &Options, seed: u64) {
    if let Some(path) = &opts.record {
        println!(
            "Not recorded to {}: only finished games are, and this one was abandoned. \
             The deal is still reproducible with --seed {seed}.",
            path.display()
        );
    }
}

/// Which game in the file this is, for the confirmation line. Cheap and best-effort: a file
/// that will not read back reports 0 rather than failing a game that was played fine.
fn record_count(path: &std::path::Path) -> usize {
    record::read_all(path).map(|games| games.len()).unwrap_or(0)
}

/// Record which of the offered actions was taken.
///
/// The **index**, not the action: an index into `legal_actions()` is all a deterministic
/// engine needs to replay a game exactly, and it costs four bytes rather than a serialised
/// `Action`. It is also the thing `netmcts` and the encoder already agree on.
fn record_choice(moves: &mut Vec<usize>, legal: &[Action], action: Action) {
    match legal.iter().position(|a| *a == action) {
        Some(index) => moves.push(index),
        // An agent that returns something it was not offered is a bug, and the caller is
        // about to `apply_trusted` it. Recording nothing would produce a file that replays
        // a *different* game and passes every check, which is far worse than stopping.
        None => panic!(
            "agent chose `{action:?}`, which was not among the {} actions it was offered",
            legal.len()
        ),
    }
}

// ================================================================== replay ==

/// One decision node, with whatever the net had to say about it.
///
/// A **node** is one decision offered to one player, which is not a turn: a turn is three
/// actions (two on the opening turn, four after an Ace), and sub-decisions — `2ND`, `NEXT`,
/// `MOVE`, `BACK`, `PEEK` — are separate zero-cost nodes on top of that. See
/// `GameState::ply` for the three counters and why they are named the way they are.
struct NodeRow {
    node: usize,
    actor: Player,
    human: bool,
    played: String,
    /// Value head for the player to move, `-1.0..=1.0`. `None` without a checkpoint.
    value: Option<f32>,
    /// The played action's share of the policy prior.
    prior_played: Option<f32>,
    /// The net's own pick, when it differs from what was played.
    net_pick: Option<(String, f32)>,
    /// What a search at `--sims` had to say, when one was run.
    search: Option<SearchView>,
}

struct SearchView {
    pick: String,
    /// The pick's share of the visits. Low on a wide branching factor even when the search
    /// is confident, so it is a spread rather than a probability.
    share: f32,
    /// The root value, converted to the value head's `-1.0..=1.0` scale so the two compare.
    value: f32,
    /// Whether the search would have played what was actually played.
    agreed: bool,
}

/// Walk a recorded game and print what the net thought at each decision — `PLAN.md` §4.0.
///
/// The instrument for §4.0a's diagnosis: sorting the nodes where the agent went wrong into
/// *(i)* moves more search fixes, *(ii)* moves it plays the same way at any budget but
/// scores wrongly — a value-function problem — and *(iii)* moves that look fine at every
/// budget and are still wrong. The third bucket is the only one no amount of compute
/// reaches, and it is why a human series is worth more than another self-play table.
///
/// The search here is a **fresh** one, not a reproduction of the search that played the
/// game: an agent's RNG stream depends on how many times it has been called, and this
/// rebuilds it. That is the right thing for analysing the *human's* decisions, which is what
/// the tool is for; the bot's own moves are in the record already.
fn cmd_replay(args: &[String]) -> Result<(), String> {
    let mut sims: Option<usize> = None;
    let mut checkpoint: Option<String> = None;
    let mut all_nodes = false;
    let mut show_node: Option<usize> = None;

    // Only the flags this command owns; the rest falls through to `parse_options`, the same
    // way `cmd_selfplay` does it. `--sims` and `--checkpoint` mean something else there,
    // which is exactly why they are peeled here rather than made global.
    //
    // `--ply` and `--all-plies` are the names these carried before the node/turn/round split
    // and are still accepted, undocumented: the muscle memory and the command lines already
    // written down in `CLAUDE.md` history are worth more than the tidiness of dropping them.
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--sims" => sims = Some(next_number(args, &mut i, "--sims")?),
            "--checkpoint" => checkpoint = Some(next_value(args, &mut i, "--checkpoint")?),
            "--node" | "--ply" => {
                let flag = args[i].clone();
                show_node = Some(next_number(args, &mut i, &flag)?);
            }
            "--all-nodes" | "--all-plies" => all_nodes = true,
            other => rest.push(other.to_string()),
        }
        i += 1;
    }
    let opts = parse_options(&rest)?;
    let path = opts
        .record
        .clone()
        .ok_or("replay needs --record <file> — the JSONL written by `play --record`")?;
    let games = record::read_all(&path)?;
    if games.is_empty() {
        return Err(format!("{} has no games in it", path.display()));
    }

    // No --game is a request for the index, not an error. It is the first thing anyone
    // wants from a file they have not opened in a month.
    let Some(number) = opts.game else {
        println!("{} — {} game(s)\n", path.display(), games.len());
        println!("{:>4}  {:>12}  {:>6}  {:>6}  {:>7}  opponent", "game", "seed", "seat", "nodes", "result");
        for (i, g) in games.iter().enumerate() {
            println!(
                "{:>4}  {:>12}  {:>6}  {:>6}  {:>7}  {}",
                i + 1,
                g.seed,
                g.human.to_string(),
                g.moves.len(),
                g.human_result,
                format!(
                    "{}{}",
                    g.opponent.as_deref().unwrap_or("hotseat"),
                    if g.hint.is_some() { "  (hinted)" } else { "" },
                ),
            );
        }
        println!("\nWalk one with --game <n>.");
        return Ok(());
    };
    let game = games
        .get(number.wrapping_sub(1))
        .ok_or_else(|| format!("--game {number} is out of range (1..={})", games.len()))?;

    // The checkpoint and budget default to the agent that was actually played, so a bare
    // `replay --game 1` says what the opponent thought. Overriding `--checkpoint` is what
    // scores an old game with a *new* net, which is §4.0a's fixed evaluation set.
    let recorded_agent = game
        .opponent
        .as_deref()
        .and_then(|spec| AgentSpec::parse(spec).ok());
    let (default_checkpoint, default_sims) = match &recorded_agent {
        Some(AgentSpec::NetMcts { checkpoint, sims }) => (Some(checkpoint.clone()), *sims),
        Some(AgentSpec::NetPolicy { checkpoint }) => (Some(checkpoint.clone()), 0),
        _ => (None, 0),
    };
    let checkpoint = checkpoint.or(default_checkpoint);
    let sims = sims.unwrap_or(default_sims);

    let mut evaluator = None;
    let mut searcher = None;
    if let Some(path) = &checkpoint {
        evaluator = Some(
            duel52_engine::nn::evaluator_for(std::path::Path::new(path), &game.config)
                .map_err(|e| format!("cannot load {path}: {e}"))?,
        );
        if sims > 0 {
            searcher = Some(duel52_engine::NetMctsAgent::derived(
                path.clone(),
                game.seed ^ 0x5EA7,
                7,
                sims,
            ));
        }
    }

    println!("{} game {} of {}", path.display(), number, games.len());
    println!(
        "  seed {} · you were {} · opponent {} · {} nodes · {}",
        game.seed,
        game.human,
        game.opponent.as_deref().unwrap_or("hotseat"),
        game.moves.len(),
        game.outcome,
    );
    // Said before the table rather than in a footnote. Every "you agreed with the net" number
    // below is worth much less when the net's pick was on screen while the human chose, and
    // the reader has to know that before they read the numbers, not after.
    if let Some(advisor) = &game.hint {
        println!("  ⚠ HINTED: `{advisor}` listed its top moves before each of your decisions.");
        println!("    Agreement below is not evidence about how you play unassisted.");
    }
    match (&checkpoint, sims) {
        (None, _) => println!(
            "  no checkpoint to score with — pass --checkpoint <file> for the net's view"
        ),
        (Some(c), 0) => println!("  scoring with {c} · policy and value only (--sims N to search)"),
        (Some(c), n) => println!("  scoring with {c} · searching {n} simulations per decision"),
    }
    println!();

    let mut obs = vec![0.0f32; duel52_engine::obs_dim(&game.config)];
    let mut logits = vec![0.0f32; duel52_engine::action_dim(&game.config)];
    let mut values = vec![0.0f32; 1];
    let mut rows: Vec<NodeRow> = Vec::new();
    let mut board: Option<String> = None;

    let final_state = game.walk(|state, legal, chosen| {
        let node = rows.len() + 1;
        let actor = state.acting_player();
        let human = game.opponent.is_none() || actor == game.human;
        // What the player could see when they chose, unless --reveal. A description built
        // from ground truth would show a face-down card's rank, which is not the decision
        // that was actually taken.
        let observer = if opts.reveal { None } else { Some(actor) };
        let action = legal[chosen];

        if show_node == Some(node) {
            board = Some(format!(
                "{}\n  node {node} · {actor}{} · played: {}\n",
                render(state, observer),
                if human { " (you)" } else { " (bot)" },
                describe_action(state, action, observer),
            ));
        }

        let mut row = NodeRow {
            node,
            actor,
            human,
            played: describe_move(state, action, observer),
            value: None,
            prior_played: None,
            net_pick: None,
            search: None,
        };

        // Only the decisions being analysed pay for a forward pass, so a 172-node game with
        // --sims 4096 costs the human's half of it rather than all of it.
        if (human || all_nodes) && evaluator.is_some() {
            let evaluator = evaluator.as_ref().expect("checked");
            duel52_engine::encode_observation(state, actor, &mut obs);
            evaluator.eval_batch(&obs, 1, &mut logits, &mut values);
            row.value = Some(values[0]);

            let priors = masked_softmax(&logits, legal, state);
            row.prior_played = Some(priors[chosen]);
            let best = priors
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, p)| (i, *p))
                .expect("a decision has at least one legal action");
            if best.0 != chosen {
                row.net_pick = Some((describe_move(state, legal[best.0], observer), best.1));
            }

            if let Some(searcher) = searcher.as_mut() {
                // Only when someone is watching: piped, `\r` does not overwrite and the
                // progress line becomes 53 lines of noise above the table.
                if io::stderr().is_terminal() {
                    eprint!("\r  searching node {node}…    ");
                    let _ = io::stderr().flush();
                }
                let result = searcher.search(state, legal);
                let total: u32 = result.visits.iter().sum();
                let top = result
                    .visits
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, v)| **v)
                    .map(|(i, v)| (i, *v))
                    .expect("a search visits at least one action");
                row.search = Some(SearchView {
                    pick: describe_move(state, legal[top.0], observer),
                    share: top.1 as f32 / total.max(1) as f32,
                    // `root_value` is a win probability in 0..=1; the value head is a tanh
                    // in -1..=1. Put both on the head's scale so a row compares.
                    value: result.root_value * 2.0 - 1.0,
                    agreed: top.0 == chosen,
                });
            }
        }
        rows.push(row);
    })?;
    if searcher.is_some() && io::stderr().is_terminal() {
        eprint!("\r                          \r");
        let _ = io::stderr().flush();
    }

    if let Some(text) = board {
        println!("{text}");
    }

    print_replay_table(&rows, all_nodes);
    print_replay_summary(&rows, &final_state, game);
    Ok(())
}

/// Softmax over the legal actions only, in the order `legal` gives them — which is the order
/// the record's indices refer to.
fn masked_softmax(logits: &[f32], legal: &[Action], state: &GameState) -> Vec<f32> {
    let picked: Vec<f32> = legal
        .iter()
        .map(|a| logits[duel52_engine::encode_action(a, state)])
        .collect();
    let max = picked.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = picked.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum.max(f32::MIN_POSITIVE)).collect()
}

fn print_replay_table(rows: &[NodeRow], all_nodes: bool) {
    println!(
        "{:>4}  {:>9}  {:>6}  {:<34} {:>6}  {}",
        "node", "actor", "value", "played", "prior", "second opinion"
    );
    for row in rows {
        if !row.human && !all_nodes {
            continue;
        }
        let value = match row.value {
            Some(v) => format!("{v:+.2}"),
            None => "—".to_string(),
        };
        let prior = match row.prior_played {
            Some(p) => format!("{p:.3}"),
            None => "—".to_string(),
        };
        // The search has the last word when it ran: it is the stronger of the two, and a
        // policy disagreement it overturned is not the interesting fact about the node.
        let alternative = match (&row.net_pick, &row.search) {
            (_, Some(s)) if s.agreed => format!(
                "search agrees ({:.0}% of visits, v {:+.2})",
                s.share * 100.0,
                s.value
            ),
            (_, Some(s)) => format!(
                "search: {} ({:.0}% of visits, v {:+.2})",
                clip(&s.pick, 40),
                s.share * 100.0,
                s.value
            ),
            (Some((pick, prior)), None) => format!("policy: {} ({prior:.3})", clip(pick, 40)),
            (None, None) => String::new(),
        };
        println!(
            "{:>4}  {:>9}  {:>6}  {:<34} {:>6}  {}",
            row.node,
            format!("{}{}", row.actor, if row.human { " you" } else { "" }),
            value,
            clip(&row.played, 34),
            prior,
            alternative,
        );
    }
}

/// The nodes §4.0a asks for: where the value head was **confident and wrong**.
///
/// Confident means `|v| > 0.6` — better than 4:1 on a `-1..=1` scale — and wrong means it
/// favoured the side that went on to lose. These are the positions to look at first, because
/// a value head that is merely uncertain is behaving correctly and a value head that is
/// confidently backing a loser is the failure everything past the search horizon inherits.
fn print_replay_summary(rows: &[NodeRow], final_state: &GameState, game: &GameRecord) {
    // Both counters, once, at the bottom: the table is indexed by node and every board it
    // prints is stamped with a turn, and the two numbers are far enough apart that seeing
    // them side by side is the shortest possible answer to "why does this skip?".
    println!(
        "\n{} decision node(s) over {} turn(s) — a turn is 3 actions (2 on the opening turn,\n\
         4 after an Ace), and sub-decisions are extra nodes that cost no action.",
        rows.len(),
        final_state.ply,
    );

    let winner = match final_state.outcome {
        Outcome::Win(w) => Some(w),
        _ => None,
    };
    let Some(winner) = winner else {
        println!("\nThe game was drawn, so there is no confidently-wrong node to point at.");
        return;
    };
    let mut wrong: Vec<&NodeRow> = Vec::new();
    for row in rows {
        if let Some(v) = row.value {
            let favours_actor = v > 0.0;
            let actor_won = row.actor == winner;
            if v.abs() > 0.6 && favours_actor != actor_won {
                wrong.push(row);
            }
        }
    }
    println!(
        "\n{} wins. Value head confident (|v| > 0.6) in the side that lost: {} of {} scored node(s).",
        winner,
        wrong.len(),
        rows.iter().filter(|r| r.value.is_some()).count(),
    );
    if wrong.is_empty() {
        return;
    }
    println!("  nodes: {}", join_nodes(&wrong));
    println!(
        "  Sort these into: (i) the search fixes it at a higher --sims, (ii) it plays the\n  \
         same at any budget but scores the position wrongly — a value-function problem, and\n  \
         (iii) it looks fine at every budget and is still wrong. Only (iii) needs a human.",
    );
    let _ = game;
}

fn join_nodes(rows: &[&NodeRow]) -> String {
    rows.iter()
        .map(|r| r.node.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Truncate to `width`, with an ellipsis, counting characters rather than bytes.
fn clip(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let mut out: String = text.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn play_help() -> String {
    "\
Choosing a move — one question at a time:
  PLAY   #1   which card from hand, then which lane
  FLIP   #2   which card, then which lane — only if that card is in more than one
  ATTACK #3   using which card, then which lane — only if that card is in more than
              one — then which of theirs
  PAIR   #4   which lane, then the two cards

  Those four numbers never move. A verb, lane or card that cannot be used right now keeps
  its place in the list and shows `—` instead of a number, so `3` is attack every turn and
  the second card in a lane is `#2` whether or not the first one can act.

  FLIP and ATTACK list your cards by rank, with `×2` when you have two of them and the
  lanes they are sitting in. A rank you have one of needs no second question. Two copies in
  one lane that are the same card in every respect are one move, so the menu takes the
  first rather than asking a question with one answer.

  A card's number counts DOWN its column on the board, so what you read off the board is
  what you type. `powers` has the full text of every power.

  The screen is redrawn after every selection: what you are choosing between is always
  directly below the board it applies to, with the last few moves above it. Run with
  --no-clear to keep every prompt in the scrollback instead.

  OR USE THE ARROW KEYS. Up and down walk the list — only the lines you could actually
  pick, so a `—` line is stepped over — right or Enter takes the line you are on, and left
  goes back one question. The line you are on carries a `*`, and the card it names lights
  up on the board exactly as a typed number would. Every question opens with the first
  line you could pick already marked, so Enter alone takes it. Typing a number and arrowing
  are the same list reached two ways; use whichever suits the move.

  TYPE A NUMBER AND LOOK UP. Before you press Enter, that line and the card it names turn
  red on the board. This is how you tell three identical `(? ²♥)` in an enemy lane apart:
  type 1, 2, 3 and watch which one lights up. Backspace or Ctrl-U takes it back; nothing is
  committed until Enter. A number no line has lights nothing up, so you know it is not one
  of the choices before you commit.

  A line lights only what it settles. `3` at the top reddens the cards that could attack;
  choosing one leaves just that card lit, and the enemy card reddens on the page that asks
  which enemy card — not before, or the list you are picking from would be red already.

  PLAY is the one step that reddens a lane heading rather than a card, because until the
  card is on the table the lane is the whole of the move.

  (The red needs a colour terminal; the `*` and the arrow keys do not. Under --no-clear, or
  piped to a file, the prompt goes back to whole typed lines with no preview and no arrows.)

Under --hint, a block below the menu:
  The moves the advising net would consider here, best first. `sims` is the share of its
  search that went into a move — the rule it picks by, so row 1 is the move it would play —
  and `after` is what it thinks your chances are once you have played it. `—` there means
  the search never went down that branch at all, which is not the same as a bad score. The
  header line is what it makes of the position before you move, so a move whose `after`
  beats it is one the net thinks improves your position.

  It is thinking from your side of the table: it samples worlds consistent with what YOU
  know, so it can no more see the opponent's hand — or your own base card — than you can.
  It never touches the game: the opponent's moves are identical whether hints are on or off,
  and the seed replays the same deal. A recorded game notes that hints were on.

Commands at the prompt:
  <number>   pick that numbered line — see it on the board first
  up / down  move to the next line you could pick; it is marked with a `*`
  right      take the line you are on, same as Enter
  left       back one question
  Ctrl-U     clear what you have typed
  0 / b      back one question
  help / ?   this message
  powers     the card-power reference
  rules      the handful of rules people get wrong
  board      redraw the screen
  q          quit

Reading the board:
  Lanes are columns, left to right. The opponent is at the top, you are at the bottom, and
  each side's base card sits at the far end of its column. The double rule across the
  middle is the front line: cards can only reach each other across it, in their own lane.

  Every card is the same width, so a column never shifts as the position changes:

  [3 ²♥]     FACE-UP 3, two hit points left. Power live, can attack.
  (3 ²♥)     FACE-DOWN, and you know it is a 3 — you played it, or a 4 showed you. No
             power, cannot attack, and a blank 2 HP whatever its rank, so a face-down Jack
             dies to two hits like anything else. Flip it and its ceiling rises to 3.
  (? ²♥)     face-down, and nobody has told you what it is
  [10³♥]     the 10 eats the space; the width does not change

  The card on the far end of a column is that side's BASE card: untouchable until every
  draw pile is empty, and while it is face-down it is hidden from you as well as from your
  opponent — which is why a 4 can usefully peek at your own.

  Two columns follow each card when there is something to say:
  a b c      member of that declared pair: attacks only together, and only a Queen or a
             death can break it
  *          FROZEN — cannot attack and cannot be flipped, by anyone, until its next turn
  ·          already attacked this turn
  +          has more than one attack left this turn (a freshly flipped Ace)
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

// ============================================================ Phase 3: the training loop ==

/// One generation of self-play. The producer half of `PLAN.md` Phase 3 step 3; the consumer
/// is `python -m duel52.train`, which calls this as a subprocess.
fn cmd_selfplay(args: &[String]) -> Result<(), String> {
    use duel52_engine::selfplay::{self, SelfPlayConfig};

    let mut checkpoint: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut generation = 0u32;
    let mut quiet = false;
    let mut sp = SelfPlayConfig::default();

    // Only the flags this command owns; everything else falls through to `parse_options`,
    // so a typo is an error rather than a silent default.
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => checkpoint = Some(next_value(args, &mut i, "--checkpoint")?),
            "--out" => out_path = Some(next_value(args, &mut i, "--out")?),
            "--generation" => generation = next_number(args, &mut i, "--generation")?,
            "--sims" => sp.sims = next_number(args, &mut i, "--sims")?,
            "--c-puct" => sp.c_puct = next_number(args, &mut i, "--c-puct")?,
            "--dirichlet-alpha" => {
                sp.noise.alpha = next_number(args, &mut i, "--dirichlet-alpha")?
            }
            "--dirichlet-weight" => {
                sp.noise.weight = next_number(args, &mut i, "--dirichlet-weight")?
            }
            "--temperature" => sp.temperature = next_number(args, &mut i, "--temperature")?,
            "--temperature-decisions" => {
                sp.temperature_decisions = next_number(args, &mut i, "--temperature-decisions")?
            }
            "--full-search-fraction" => {
                sp.full_search_fraction = next_number(args, &mut i, "--full-search-fraction")?
            }
            "--cap-sims" => sp.cap_sims = next_number(args, &mut i, "--cap-sims")?,
            "--quiet" => quiet = true,
            other => rest.push(other.to_string()),
        }
        i += 1;
    }
    let opts = parse_options(&rest)?;
    let checkpoint = checkpoint.ok_or("selfplay needs --checkpoint <path>")?;
    let out_path = out_path.ok_or("selfplay needs --out <path>")?;
    let games = opts.games_or(1000);
    if sp.sims == 0 {
        return Err("--sims must be at least 1".to_string());
    }
    if !(0.0..=1.0).contains(&sp.full_search_fraction) {
        return Err("--full-search-fraction must be between 0 and 1".to_string());
    }
    if sp.full_search_fraction < 1.0 {
        if sp.cap_sims == 0 {
            return Err("--cap-sims must be at least 1".to_string());
        }
        if sp.cap_sims >= sp.sims {
            // Not a hard requirement of the format, but it is always a mistake: a cap that
            // is not smaller than the full budget buys no time and costs policy targets.
            return Err(format!(
                "--cap-sims {} must be smaller than --sims {} to be worth anything",
                sp.cap_sims, sp.sims
            ));
        }
    }

    let out = std::path::PathBuf::from(&out_path);
    let report = selfplay::run(
        opts.config,
        &sp,
        std::path::Path::new(&checkpoint),
        opts.seed,
        games,
        opts.threads,
        opts.eval_batch,
        generation,
        &out,
        !quiet,
    )?;
    print!("{}", report.report(&out));
    if report.max_slots_seen > opts.config.encoding_slots {
        // Unreachable — the encoder asserts first — but if the assertion is ever relaxed
        // this is the line that says the corpus is compromised rather than merely odd.
        eprintln!(
            "warning: a lane side held {} cards, above --encoding-slots {}",
            report.max_slots_seen, opts.config.encoding_slots
        );
    }
    Ok(())
}

/// Print a shard's header and replay it, which is the integrity check that matters: a
/// trajectory is only worth keeping if the engine still produces the same legal-action lists
/// it was recorded against.
fn cmd_shard(args: &[String]) -> Result<(), String> {
    use duel52_engine::selfplay;

    let path = args.first().ok_or("shard needs a file path")?;
    let threads = if let Some(pos) = args.iter().position(|a| a == "--threads") {
        args.get(pos + 1)
            .and_then(|v| v.parse().ok())
            .ok_or("--threads needs a number")?
    } else {
        default_threads()
    };

    let shard = selfplay::Shard::read(std::path::Path::new(path))?;
    println!("{path}");
    for (k, v) in &shard.header {
        println!("  {k:<22} {v}");
    }
    println!("  {:<22} {}", "games", shard.games.len());
    println!("  {:<22} {}", "samples", shard.sample_count());
    println!("  {:<22} {}", "config", shard.config.summary());

    let started = std::time::Instant::now();
    let set = duel52_engine::selfplay::replay(&shard, threads, 1);
    let secs = started.elapsed().as_secs_f64();
    if set.samples != shard.sample_count() {
        return Err(format!(
            "replay produced {} samples but the shard holds {} — the recorded trajectories \
             do not match this engine build",
            set.samples,
            shard.sample_count()
        ));
    }
    println!(
        "\nreplayed cleanly in {secs:.1}s — {} samples, obs_dim {}, action_dim {}",
        set.samples, set.obs_dim, set.action_dim
    );
    println!(
        "  observation density {:.1}% ({:.0} of {} features per sample)",
        100.0 * set.obs_index.len() as f64 / (set.samples * set.obs_dim).max(1) as f64,
        set.obs_index.len() as f64 / set.samples.max(1) as f64,
        set.obs_dim,
    );
    println!(
        "  policy target support {:.1} actions per sample",
        set.policy_index.len() as f64 / set.samples.max(1) as f64
    );
    let mean_value = set.value.iter().sum::<f32>() / set.samples.max(1) as f32;
    let mean_root = set.root_value.iter().sum::<f32>() / set.samples.max(1) as f32;
    println!("  mean value target {mean_value:+.4} · mean search root value {mean_root:+.4}");
    Ok(())
}
