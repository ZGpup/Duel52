//! Resumable jobs: an append-only record of finished games, so a job killed part way through
//! starts again from where it was rather than from nothing.
//!
//! # Why this exists
//!
//! The training loop ran on a shared 128-core box that pauses jobs by killing them. A
//! generation there is ~32 minutes, ~21 of them one `duel52 selfplay` process that held every
//! game in memory until the end — so a pause at minute 20 cost all 20. The same was true of
//! every gate and panel match. `python -m duel52.train run --resume` could only restart at a
//! generation boundary, because nothing smaller than a generation was ever on disk.
//!
//! # The unit of saved work is one finished game
//!
//! Every game in this engine is a pure function of `(config, agents, seed)`, and every job is a
//! set of games indexed by seed. So a game that finished is finished for good, and a job can be
//! resumed by skipping the indices already on disk. Nothing about a game *in flight* is saved:
//! a half-played search is megabytes of tree, and at 128 cores the in-flight games are worth
//! about a minute and a half of self-play, which is the floor a pause costs whatever else is done.
//!
//! **A resumed job produces exactly what an uninterrupted one would.** Self-play sorts its games
//! by seed before writing, so the shard is byte-identical
//! (`phase4_a_resumed_shard_is_byte_identical_to_an_uninterrupted_one`).
//!
//! # The format
//!
//! ```text
//! magic "D52JN\0" · version u16 · fingerprint length u32 · fingerprint (UTF-8)
//! then per record:  index u64 · payload length u32 · payload · check u64
//! ```
//!
//! `check` is FNV-1a over the index, the length and the payload. A process killed mid-write
//! leaves a short last record, and a machine that lost power can leave garbage past its last
//! sync; both are caught the same way. Reading stops at the first record that does not verify
//! and the file is truncated there, so the next append follows the last good record.
//!
//! **The fingerprint is what stops a journal resuming the wrong job.** It is the caller's whole
//! description of the work: seeds, game count, search settings, the ruleset, and a hash of the
//! *contents* of every checkpoint involved. Content and not path, because the training loop
//! plays every generation's self-play from the same `best.d52nn`. A journal whose fingerprint
//! differs is discarded and started over, and the caller says so.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const JOURNAL_MAGIC: &[u8; 6] = b"D52JN\0";
pub const JOURNAL_VERSION: u16 = 1;

/// How often an append also forces the file to stable storage.
///
/// Every record is handed to the operating system the moment its game finishes, and that alone
/// survives the process being killed, which is what a pause is. A sync is only for the machine
/// itself going down. Syncing per game would put a disk flush under a lock the workers share,
/// 38 times a second at 128 cores and far worse on network storage, for protection against a
/// failure that already costs the in-flight games. Thirty seconds is well under that floor.
pub const SYNC_EVERY: Duration = Duration::from_secs(30);

/// An open journal. Appends are safe from any number of threads.
pub struct Journal {
    path: PathBuf,
    inner: Mutex<Inner>,
}

struct Inner {
    file: File,
    last_sync: Instant,
    /// The first write that failed. After one, the journal stops writing and says so from
    /// [`Journal::finish`], rather than failing the job — the job's output is still correct,
    /// it just cannot be resumed.
    error: Option<String>,
}

/// What [`Journal::open`] found.
pub struct Opened {
    pub journal: Journal,
    /// Records that survived an earlier attempt: the first occurrence of each index, in the
    /// order they were written.
    pub restored: Vec<(u64, Vec<u8>)>,
    /// Why a journal that was already on disk was started over, if it was.
    pub discarded: Option<String>,
}

impl Journal {
    /// Open `path` for the job described by `fingerprint`, resuming it if the file on disk is
    /// the same job, and starting it fresh otherwise.
    pub fn open(path: &Path, fingerprint: &str) -> Result<Opened, String> {
        let io = |e: std::io::Error| format!("journal `{}`: {e}", path.display());
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(io)?;
            }
        }

        let mut discarded = None;
        if path.exists() {
            let mut file = OpenOptions::new().read(true).write(true).open(path).map_err(io)?;
            match read_existing(&mut file, fingerprint).map_err(io)? {
                Existing::Resume { records, good_len } => {
                    // Anything after the last record that verifies is a torn write. Cut it, so
                    // the next append does not bury a good record behind a bad one.
                    file.set_len(good_len).map_err(io)?;
                    file.seek(SeekFrom::End(0)).map_err(io)?;
                    return Ok(Opened {
                        journal: Journal::wrap(path, file),
                        restored: records,
                        discarded: None,
                    });
                }
                Existing::Discard(why) => discarded = Some(why),
            }
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(io)?;
        let mut header = Vec::with_capacity(12 + fingerprint.len());
        header.extend_from_slice(JOURNAL_MAGIC);
        header.extend_from_slice(&JOURNAL_VERSION.to_le_bytes());
        header.extend_from_slice(&(fingerprint.len() as u32).to_le_bytes());
        header.extend_from_slice(fingerprint.as_bytes());
        file.write_all(&header).map_err(io)?;
        file.sync_data().map_err(io)?;
        Ok(Opened {
            journal: Journal::wrap(path, file),
            restored: Vec::new(),
            discarded,
        })
    }

    fn wrap(path: &Path, file: File) -> Journal {
        Journal {
            path: path.to_path_buf(),
            inner: Mutex::new(Inner {
                file,
                last_sync: Instant::now(),
                error: None,
            }),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Record one finished unit of work. One `write` call per record, so the operating system
    /// has the whole record before this returns.
    pub fn append(&self, index: u64, payload: &[u8]) {
        let mut record = Vec::with_capacity(20 + payload.len());
        record.extend_from_slice(&index.to_le_bytes());
        record.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        record.extend_from_slice(payload);
        record.extend_from_slice(&check(index, payload).to_le_bytes());

        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if inner.error.is_some() {
            return;
        }
        let result = inner.file.write_all(&record).and_then(|()| {
            if inner.last_sync.elapsed() >= SYNC_EVERY {
                inner.file.sync_data()?;
                inner.last_sync = Instant::now();
            }
            Ok(())
        });
        if let Err(e) = result {
            inner.error = Some(format!("journal `{}`: {e}", self.path.display()));
        }
    }

    /// Sync what has been written. `Err` if any append failed along the way, in which case the
    /// journal is incomplete and a resume would redo the work it is missing.
    pub fn finish(&self) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(e) = inner.error.take() {
            return Err(e);
        }
        inner
            .file
            .sync_data()
            .map_err(|e| format!("journal `{}`: {e}", self.path.display()))
    }
}

enum Existing {
    Resume {
        records: Vec<(u64, Vec<u8>)>,
        good_len: u64,
    },
    Discard(String),
}

fn read_existing(file: &mut File, fingerprint: &str) -> std::io::Result<Existing> {
    let total = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(1 << 20, &mut *file);

    let mut head = [0u8; 12];
    if reader.read_exact(&mut head).is_err() || &head[..6] != JOURNAL_MAGIC {
        return Ok(Existing::Discard("it is not a journal, or its header is torn".into()));
    }
    let version = u16::from_le_bytes([head[6], head[7]]);
    if version != JOURNAL_VERSION {
        return Ok(Existing::Discard(format!(
            "it is journal version {version} and this build writes {JOURNAL_VERSION}"
        )));
    }
    let len = u32::from_le_bytes(head[8..12].try_into().expect("4 bytes")) as u64;
    if 12 + len > total {
        return Ok(Existing::Discard("its header is torn".into()));
    }
    let mut recorded = vec![0u8; len as usize];
    reader.read_exact(&mut recorded)?;
    if recorded != fingerprint.as_bytes() {
        return Ok(Existing::Discard(
            "it was written for a different job (seed, settings, ruleset or checkpoint contents \
             differ)"
                .into(),
        ));
    }

    let mut good_len = 12 + len;
    let mut records: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    loop {
        let mut fixed = [0u8; 12];
        if good_len + 12 > total || reader.read_exact(&mut fixed).is_err() {
            break;
        }
        let index = u64::from_le_bytes(fixed[..8].try_into().expect("8 bytes"));
        let size = u32::from_le_bytes(fixed[8..].try_into().expect("4 bytes")) as u64;
        // Checked against the file before allocating, so a torn length cannot ask for gigabytes.
        if good_len + 12 + size + 8 > total {
            break;
        }
        let mut payload = vec![0u8; size as usize];
        let mut tail = [0u8; 8];
        if reader.read_exact(&mut payload).is_err() || reader.read_exact(&mut tail).is_err() {
            break;
        }
        if u64::from_le_bytes(tail) != check(index, &payload) {
            break;
        }
        good_len += 12 + size + 8;
        if seen.insert(index) {
            records.push((index, payload));
        }
    }
    Ok(Existing::Resume { records, good_len })
}

fn check(index: u64, payload: &[u8]) -> u64 {
    let mut bytes = Vec::with_capacity(12 + payload.len());
    bytes.extend_from_slice(&index.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(payload);
    crate::encode::fnv1a64(&bytes)
}

/// FNV-1a of a file's contents, for a fingerprint that names a checkpoint by what is in it.
pub fn file_hash(path: &Path) -> Result<u64, String> {
    std::fs::read(path)
        .map(|bytes| crate::encode::fnv1a64(&bytes))
        .map_err(|e| format!("cannot read `{}`: {e}", path.display()))
}

/// Write `bytes` to `path` so that a reader sees the old file or the new one and never half of
/// either: into a sibling, synced, then renamed over.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let io = |e: std::io::Error| format!("cannot write `{}`: {e}", path.display());
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
    }
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".tmp");
    let tmp = path.with_file_name(name);
    let mut file = File::create(&tmp).map_err(io)?;
    file.write_all(bytes).map_err(io)?;
    file.sync_data().map_err(io)?;
    drop(file);
    std::fs::rename(&tmp, path).map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("duel52-journal-{}-{tag}.journal", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn write(path: &Path, fingerprint: &str, records: &[(u64, &[u8])]) {
        let opened = Journal::open(path, fingerprint).expect("open");
        for &(i, p) in records {
            opened.journal.append(i, p);
        }
        opened.journal.finish().expect("finish");
    }

    #[test]
    fn what_was_appended_is_what_is_restored() {
        let path = temp("roundtrip");
        write(&path, "job", &[(3, b"three"), (0, b""), (7, &[9u8; 300])]);
        let opened = Journal::open(&path, "job").expect("reopen");
        assert!(opened.discarded.is_none());
        assert_eq!(
            opened.restored,
            vec![(3, b"three".to_vec()), (0, Vec::new()), (7, vec![9u8; 300])]
        );
    }

    #[test]
    fn a_torn_last_record_is_dropped_and_the_next_append_follows_the_good_ones() {
        let path = temp("torn");
        write(&path, "job", &[(1, b"one"), (2, b"two")]);
        let full = std::fs::metadata(&path).unwrap().len();
        // A kill in the middle of writing record 2.
        OpenOptions::new().write(true).open(&path).unwrap().set_len(full - 5).unwrap();

        let opened = Journal::open(&path, "job").expect("reopen");
        assert_eq!(opened.restored, vec![(1, b"one".to_vec())]);
        opened.journal.append(2, b"two again");
        opened.journal.finish().unwrap();

        let again = Journal::open(&path, "job").expect("third open");
        assert_eq!(again.restored, vec![(1, b"one".to_vec()), (2, b"two again".to_vec())]);
    }

    #[test]
    fn a_record_that_does_not_verify_ends_the_restore() {
        let path = temp("corrupt");
        write(&path, "job", &[(1, b"one"), (2, b"two"), (3, b"three")]);
        let mut bytes = std::fs::read(&path).unwrap();
        // The payload of record 2: header 12 + "job" 3, record 1 is 12 + 3 + 8, then 12.
        let at = 12 + 3 + 23 + 12;
        bytes[at] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();

        let opened = Journal::open(&path, "job").expect("reopen");
        assert_eq!(opened.restored, vec![(1, b"one".to_vec())], "garbage is not a game");
    }

    #[test]
    fn a_journal_for_a_different_job_is_started_over() {
        let path = temp("other");
        write(&path, "seed=1", &[(1, b"one")]);
        let opened = Journal::open(&path, "seed=2").expect("reopen");
        assert!(opened.restored.is_empty());
        assert!(opened.discarded.is_some());
        drop(opened);
        // And it is now the new job's journal, not a copy of the old one.
        assert!(Journal::open(&path, "seed=2").unwrap().discarded.is_none());

        std::fs::write(&path, b"not a journal").unwrap();
        assert!(Journal::open(&path, "seed=2").unwrap().discarded.is_some());
    }

    #[test]
    fn a_repeated_index_keeps_its_first_record() {
        let path = temp("repeat");
        write(&path, "job", &[(4, b"first"), (4, b"second")]);
        let opened = Journal::open(&path, "job").expect("reopen");
        assert_eq!(opened.restored, vec![(4, b"first".to_vec())]);
    }
}
