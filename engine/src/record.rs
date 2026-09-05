//! Recording a played game, and replaying it exactly — `PLAN.md` §4.0.
//!
//! # Why this is nearly free
//!
//! A Duel 52 game is fully determined by `(config, seed, the sequence of chosen indices into
//! [`GameState::legal_actions`])`. The engine is deterministic — same seed, same config,
//! same deal, hidden information included — so a few hundred bytes replay a game *exactly*,
//! including both hands, both base cards and the order of the draw pile. That is the same
//! insight the `.d52sp` trajectory format is built on.
//!
//! The occasion for it: the project owner beat `gen016` 5–0 and **not one ply was written
//! down**. No seeds, no moves, no way to ask what the net was thinking when it played the
//! moves that lost. The human series is the only external measurement this project has
//! (everything else is scored against agents it wrote itself, on a ladder anchored at
//! `random`), and it was not being kept.
//!
//! # Why JSONL and not a shard
//!
//! `PLAN.md` §4.0 is explicit, and the reason is a safety property rather than a taste:
//! **a `.d52sp` row must carry a policy target**, because the trainer reads shards and a row
//! without one is a row it could silently train on. A human decision has no visit
//! distribution and no root value. Keeping human games in a different, inspectable,
//! git-committable format means the corpus format stays strict and there is no shard on disk
//! that the trainer must be trusted to skip.
//!
//! One line per game, appended. A few kilobytes for a whole series, diffable, and readable
//! from Python with `json.loads` for the analysis in §4.0a.
//!
//! # What is checked on the way back in
//!
//! [`GameRecord::walk`] is not a decoder, it is a *verifier*. It replays the moves against a
//! fresh engine and refuses the record unless every index was in range, the game ended
//! exactly when the moves ran out, and the outcome matches the one written down. So a record
//! that no longer reproduces — because the rules changed, or the config string means
//! something different now — fails loudly rather than quietly describing a different game.
//! That is the same argument as the checkpoint header's layout hashes.

use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::Path;

use crate::action::Action;
use crate::config::GameConfig;
use crate::outcome::Outcome;
use crate::player::Player;
use crate::state::GameState;
use crate::VERSION;

/// The format tag written into every line, and the only one [`GameRecord::parse`] accepts.
///
/// Bump the number if the meaning of a field changes. A reader that finds a tag it does not
/// know refuses the line, which is preferable to interpreting `moves` under the wrong rules.
pub const FORMAT: &str = "duel52-play/1";

/// One completed game, as played.
#[derive(Clone, Debug, PartialEq)]
pub struct GameRecord {
    /// Engine version that produced the record. Informational — the outcome check in
    /// [`GameRecord::walk`] is what actually catches a build that plays a different game.
    pub engine: String,
    /// Unix seconds. No date formatting, because that would need a dependency and the
    /// engine has none; `date -r <n>` reads it.
    pub recorded: u64,
    pub config: GameConfig,
    pub seed: u64,
    /// The seat the human took. In a hotseat game (`opponent` is `None`) both seats are
    /// human and this is the seat that moved first.
    pub human: Player,
    /// The opponent's [`crate::AgentSpec`] string, or `None` for hotseat.
    pub opponent: Option<String>,
    /// Chosen indices into `legal_actions()`, in ply order, both sides.
    pub moves: Vec<usize>,
    /// The outcome as [`Outcome`] renders it. Checked on replay.
    pub outcome: String,
    /// `"win"`, `"loss"` or `"draw"`, from `human`'s seat. Redundant with `outcome` and
    /// kept because it is the field anyone actually greps for.
    pub human_result: String,
}

impl GameRecord {
    /// Build a record from a finished game.
    ///
    /// `moves` must be the indices chosen at each decision, in order, or the record will not
    /// verify — which is the point of writing it this way round rather than trusting it.
    pub fn new(
        config: GameConfig,
        seed: u64,
        human: Player,
        opponent: Option<String>,
        moves: Vec<usize>,
        outcome: Outcome,
    ) -> GameRecord {
        GameRecord {
            engine: VERSION.to_string(),
            recorded: unix_seconds(),
            config,
            seed,
            human,
            opponent,
            moves,
            outcome: outcome.to_string(),
            human_result: match outcome {
                Outcome::Win(w) if w == human => "win",
                Outcome::Win(_) => "loss",
                Outcome::Draw(_) => "draw",
                Outcome::Ongoing => "unfinished",
            }
            .to_string(),
        }
    }

    /// Replay the game, calling `visit` before each move with the position, the legal
    /// actions in the order the record's indices refer to, and the index chosen.
    ///
    /// Returns the final state. Every failure mode is an error rather than a panic, because
    /// the likely cause is a stale record rather than a bug: an index out of range, a game
    /// that ended early or late, or an outcome that no longer matches.
    pub fn walk(
        &self,
        mut visit: impl FnMut(&GameState, &[Action], usize),
    ) -> Result<GameState, String> {
        let mut state = GameState::new(self.config, self.seed);
        for (ply, &choice) in self.moves.iter().enumerate() {
            if state.outcome.is_over() {
                return Err(format!(
                    "the game was over after {ply} move(s) but the record has {}. The rules \
                     or the config have changed since it was played.",
                    self.moves.len()
                ));
            }
            let legal = state.legal_actions();
            let action = *legal.get(choice).ok_or_else(|| {
                format!(
                    "ply {}: the record chose index {choice} of {} legal action(s). The \
                     action list has changed shape since this game was played.",
                    ply + 1,
                    legal.len()
                )
            })?;
            visit(&state, &legal, choice);
            state.apply_trusted(action);
        }
        if !state.outcome.is_over() {
            return Err(format!(
                "the record's {} move(s) ran out with the game still in progress. The rules \
                 or the config have changed since it was played.",
                self.moves.len()
            ));
        }
        if state.outcome.to_string() != self.outcome {
            return Err(format!(
                "replayed to `{}` but the record says `{}` — this build does not reproduce \
                 the game that was played.",
                state.outcome, self.outcome
            ));
        }
        Ok(state)
    }

    /// One JSON object on one line, newline-terminated.
    pub fn to_json_line(&self) -> String {
        let mut out = String::with_capacity(512 + self.moves.len() * 4);
        out.push('{');
        write_str_field(&mut out, "format", FORMAT, true);
        write_str_field(&mut out, "engine", &self.engine, false);
        let _ = write!(out, ",\"recorded\":{}", self.recorded);
        let _ = write!(out, ",\"seed\":{}", self.seed);
        write_str_field(&mut out, "human", &self.human.to_string(), false);
        match &self.opponent {
            Some(spec) => write_str_field(&mut out, "opponent", spec, false),
            None => out.push_str(",\"opponent\":null"),
        }
        write_str_field(&mut out, "outcome", &self.outcome, false);
        write_str_field(&mut out, "human_result", &self.human_result, false);
        let _ = write!(out, ",\"plies\":{}", self.moves.len());
        out.push_str(",\"moves\":[");
        for (i, m) in self.moves.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "{m}");
        }
        out.push(']');
        // Last because it is by far the longest, so the interesting fields stay visible at
        // the head of the line when the file is read in a terminal.
        write_str_field(&mut out, "config", &self.config.to_config_string(), false);
        out.push_str("}\n");
        out
    }

    /// Parse one line written by [`GameRecord::to_json_line`].
    pub fn parse(line: &str) -> Result<GameRecord, String> {
        let value = Json::parse(line)?;
        let format = value.field_str("format")?;
        if format != FORMAT {
            return Err(format!(
                "this is a `{format}` record and this build reads `{FORMAT}`"
            ));
        }
        let config = GameConfig::from_config_str(&value.field_str("config")?)
            .map_err(|e| format!("the record's config does not load: {e}"))?;
        let human = match value.field_str("human")?.as_str() {
            "P0" => Player::P0,
            "P1" => Player::P1,
            other => return Err(format!("`human` is `{other}`, expected P0 or P1")),
        };
        let opponent = match value.field("opponent") {
            Some(Json::Null) | None => None,
            Some(Json::Str(s)) => Some(s.clone()),
            Some(_) => return Err("`opponent` must be a string or null".to_string()),
        };
        let moves = match value.field("moves") {
            Some(Json::Arr(items)) => items
                .iter()
                .map(|item| match item {
                    Json::Num(text) => text
                        .parse::<usize>()
                        .map_err(|_| format!("`{text}` is not an action index")),
                    _ => Err("`moves` must be an array of numbers".to_string()),
                })
                .collect::<Result<Vec<usize>, String>>()?,
            _ => return Err("`moves` is missing or is not an array".to_string()),
        };
        Ok(GameRecord {
            engine: value.field_str("engine").unwrap_or_default(),
            recorded: value.field_u64("recorded").unwrap_or(0),
            config,
            seed: value.field_u64("seed")?,
            human,
            opponent,
            moves,
            outcome: value.field_str("outcome")?,
            human_result: value.field_str("human_result").unwrap_or_default(),
        })
    }

    /// Append to a JSONL file, creating it and its directory if needed.
    ///
    /// Opened, written and closed per game, so a series survives the terminal being closed
    /// between games — which is how a series is actually played.
    pub fn append_to(&self, path: &Path) -> Result<(), String> {
        create_parent(path)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("cannot open {} for appending: {e}", path.display()))?;
        file.write_all(self.to_json_line().as_bytes())
            .map_err(|e| format!("cannot write to {}: {e}", path.display()))
    }
}

/// Prove a record path is writable **before** a game is played.
///
/// The whole feature exists so that games are not lost, and a path that cannot be written is
/// only discovered when the game ends — an hour after the mistake, with the game as the
/// price. `--record games/x.jsonl` with no `games/` directory did exactly that. So the
/// directory is created and the file is opened for append at start-up, which answers "can
/// this be written" while it still costs nothing.
///
/// Creating an empty file for a game that is then abandoned is harmless: [`read_all`] skips
/// blank lines, and an empty corpus reads as no games.
pub fn prepare(path: &Path) -> Result<(), String> {
    create_parent(path)?;
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("cannot write to {}: {e}", path.display()))?;
    Ok(())
}

fn create_parent(path: &Path) -> Result<(), String> {
    match path.parent() {
        // `Path::parent` of a bare filename is `""`, which `create_dir_all` would reject.
        Some(dir) if !dir.as_os_str().is_empty() => std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display())),
        _ => Ok(()),
    }
}

/// Read every record in a JSONL file, in file order.
///
/// Blank lines are skipped so a hand-edited file is still readable; anything else that will
/// not parse is an error naming its line number, because a corpus that silently drops games
/// is worse than one that will not open.
pub fn read_all(path: &Path) -> Result<Vec<GameRecord>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            GameRecord::parse(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?,
        );
    }
    Ok(out)
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn write_str_field(out: &mut String, key: &str, value: &str, first: bool) {
    if !first {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    write_json_string(out, value);
}

fn write_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ============================================================================= json ==

/// A JSON value.
///
/// The engine has no dependencies (see the workspace `Cargo.toml`: third-party RNGs do not
/// promise stability across versions, and reproducibility is a rules requirement), so this
/// is a hand-rolled reader. It is small because it only ever reads files this module wrote —
/// but it is a real parser rather than a pattern match on the writer's output, because the
/// whole point of a plain-text corpus is that a human can edit it.
///
/// Numbers are kept as their source text and parsed on demand. A `u64` seed above 2^53 does
/// not survive a round trip through `f64`, and silently changing the seed would change the
/// deal.
#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn parse(text: &str) -> Result<Json, String> {
        let bytes: Vec<char> = text.chars().collect();
        let mut at = 0usize;
        let value = parse_value(&bytes, &mut at)?;
        skip_space(&bytes, &mut at);
        if at != bytes.len() {
            return Err(format!("trailing text after the JSON value at char {at}"));
        }
        Ok(value)
    }

    fn field(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn field_str(&self, key: &str) -> Result<String, String> {
        match self.field(key) {
            Some(Json::Str(s)) => Ok(s.clone()),
            Some(_) => Err(format!("`{key}` is not a string")),
            None => Err(format!("`{key}` is missing")),
        }
    }

    fn field_u64(&self, key: &str) -> Result<u64, String> {
        match self.field(key) {
            Some(Json::Num(text)) => text
                .parse::<u64>()
                .map_err(|_| format!("`{key}`: `{text}` is not a whole number")),
            Some(_) => Err(format!("`{key}` is not a number")),
            None => Err(format!("`{key}` is missing")),
        }
    }
}

fn skip_space(text: &[char], at: &mut usize) {
    while matches!(text.get(*at), Some(' ' | '\t' | '\n' | '\r')) {
        *at += 1;
    }
}

fn parse_value(text: &[char], at: &mut usize) -> Result<Json, String> {
    skip_space(text, at);
    match text.get(*at) {
        None => Err("unexpected end of input".to_string()),
        Some('{') => parse_object(text, at),
        Some('[') => parse_array(text, at),
        Some('"') => Ok(Json::Str(parse_string(text, at)?)),
        Some('t') => parse_literal(text, at, "true", Json::Bool(true)),
        Some('f') => parse_literal(text, at, "false", Json::Bool(false)),
        Some('n') => parse_literal(text, at, "null", Json::Null),
        Some(c) if *c == '-' || c.is_ascii_digit() => parse_number(text, at),
        Some(c) => Err(format!("unexpected `{c}` at char {at}")),
    }
}

fn parse_literal(text: &[char], at: &mut usize, word: &str, value: Json) -> Result<Json, String> {
    for expected in word.chars() {
        if text.get(*at) != Some(&expected) {
            return Err(format!("expected `{word}` at char {at}"));
        }
        *at += 1;
    }
    Ok(value)
}

fn parse_number(text: &[char], at: &mut usize) -> Result<Json, String> {
    let start = *at;
    if text.get(*at) == Some(&'-') {
        *at += 1;
    }
    while matches!(text.get(*at), Some(c) if c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
    {
        *at += 1;
    }
    if *at == start {
        return Err(format!("expected a number at char {start}"));
    }
    Ok(Json::Num(text[start..*at].iter().collect()))
}

fn parse_string(text: &[char], at: &mut usize) -> Result<String, String> {
    if text.get(*at) != Some(&'"') {
        return Err(format!("expected a string at char {at}"));
    }
    *at += 1;
    let mut out = String::new();
    loop {
        match text.get(*at) {
            None => return Err("unterminated string".to_string()),
            Some('"') => {
                *at += 1;
                return Ok(out);
            }
            Some('\\') => {
                *at += 1;
                let escape = *text.get(*at).ok_or("unterminated escape")?;
                *at += 1;
                match escape {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let digit = text
                                .get(*at)
                                .and_then(|c| c.to_digit(16))
                                .ok_or_else(|| format!("bad \\u escape at char {at}"))?;
                            code = code * 16 + digit;
                            *at += 1;
                        }
                        out.push(
                            char::from_u32(code)
                                .ok_or_else(|| format!("\\u{code:04x} is not a character"))?,
                        );
                    }
                    other => return Err(format!("unknown escape `\\{other}`")),
                }
            }
            Some(c) => {
                out.push(*c);
                *at += 1;
            }
        }
    }
}

fn parse_array(text: &[char], at: &mut usize) -> Result<Json, String> {
    *at += 1; // '['
    let mut items = Vec::new();
    skip_space(text, at);
    if text.get(*at) == Some(&']') {
        *at += 1;
        return Ok(Json::Arr(items));
    }
    loop {
        items.push(parse_value(text, at)?);
        skip_space(text, at);
        match text.get(*at) {
            Some(',') => *at += 1,
            Some(']') => {
                *at += 1;
                return Ok(Json::Arr(items));
            }
            _ => return Err(format!("expected `,` or `]` at char {at}")),
        }
    }
}

fn parse_object(text: &[char], at: &mut usize) -> Result<Json, String> {
    *at += 1; // '{'
    let mut fields = Vec::new();
    skip_space(text, at);
    if text.get(*at) == Some(&'}') {
        *at += 1;
        return Ok(Json::Obj(fields));
    }
    loop {
        skip_space(text, at);
        let key = parse_string(text, at)?;
        skip_space(text, at);
        if text.get(*at) != Some(&':') {
            return Err(format!("expected `:` after `{key}` at char {at}"));
        }
        *at += 1;
        fields.push((key, parse_value(text, at)?));
        skip_space(text, at);
        match text.get(*at) {
            Some(',') => *at += 1,
            Some('}') => {
                *at += 1;
                return Ok(Json::Obj(fields));
            }
            _ => return Err(format!("expected `,` or `}}` at char {at}")),
        }
    }
}
