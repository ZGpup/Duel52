//! Fitting Elo ratings to a round-robin result table.
//!
//! `PLAN.md` Phase 2 asks for a "round-robin Elo ladder, frozen as the permanent benchmark",
//! and Phase 3 for "an Elo table and an LBR number". This module is the arithmetic behind
//! both.
//!
//! # Not the incremental Elo update
//!
//! The familiar `R += K(S − E)` rule is an *online* estimator: it depends on the order games
//! were played in and never converges. A benchmark ladder is a batch problem, so this fits
//! the underlying Bradley–Terry model by maximum likelihood instead — the ratings that make
//! the observed result table most probable. The output is order-independent and
//! reproducible, which `CLAUDE.md` requires of anything that lands in `FINDINGS.md`.
//!
//! Draws are counted as half a point on each side, the usual convention. Duel 52's draws are
//! rare enough (`FINDINGS.md` F1.3: 0.4–0.5% under random play) that a full Davidson
//! draw-aware model would not change a rating by a point.
//!
//! # The prior, and why there is one
//!
//! A rung that wins every single game has an infinite maximum-likelihood rating. That is a
//! true statement about the likelihood and a useless one for a table, so the fit adds a weak
//! prior: [`PRIOR_GAMES`] virtual games, split evenly, against a phantom opponent at the
//! anchor rating. It pulls a perfect record in to a finite number and is negligible against
//! a few hundred real games. Any rating whose value depends on this prior is reported with a
//! standard error wide enough to say so.

use std::fmt;

/// The head-to-head record between two entrants.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Pairing {
    /// Index into the roster.
    pub a: usize,
    /// Index into the roster.
    pub b: usize,
    /// Games played between them, both colours combined.
    pub games: f64,
    /// Points scored by `a`: 1 per win, 0.5 per draw.
    pub score_a: f64,
}

impl Pairing {
    pub fn new(a: usize, b: usize, wins_a: usize, wins_b: usize, draws: usize) -> Pairing {
        Pairing {
            a,
            b,
            games: (wins_a + wins_b + draws) as f64,
            score_a: wins_a as f64 + 0.5 * draws as f64,
        }
    }
}

/// Virtual games added per entrant, split evenly, against a phantom at the anchor rating.
/// See the module docs.
pub const PRIOR_GAMES: f64 = 2.0;

/// Elo's scale constant: a 400-point gap is a 10:1 expected-score ratio.
const SCALE: f64 = 400.0;

/// A fitted rating table.
#[derive(Clone, Debug)]
pub struct EloTable {
    pub names: Vec<String>,
    pub ratings: Vec<f64>,
    /// Marginal standard error on each rating, from the diagonal of the Fisher information.
    ///
    /// It ignores the correlations between entrants, so it is the right error bar for "how
    /// well is *this* rung pinned down" and an underestimate for "is rung `i` above rung
    /// `j`". For that question use [`EloTable::expected_score`] against the head-to-head
    /// count, which is what the data actually measured.
    pub stderr: Vec<f64>,
    /// Which entrant is pinned to 0. Elo is only defined up to an additive constant.
    pub anchor: usize,
}

/// Expected score for a player rated `r_a` against one rated `r_b`.
#[inline]
pub fn expected_score(r_a: f64, r_b: f64) -> f64 {
    1.0 / (1.0 + 10f64.powf((r_b - r_a) / SCALE))
}

/// Fit ratings by maximum likelihood, anchoring `anchor` at 0.
///
/// Coordinate-wise Newton on the Bradley–Terry log-likelihood. The likelihood is concave in
/// the ratings, so there is one optimum and no starting-point sensitivity; the loop stops
/// when the largest single-entrant move falls below a tenth of an Elo point.
pub fn fit(names: Vec<String>, pairings: &[Pairing], anchor: usize) -> EloTable {
    let n = names.len();
    assert!(anchor < n, "anchor {anchor} is not in a roster of {n}");
    let mut ratings = vec![0.0f64; n];
    // Natural-log form of the 400-point scale, which is what the derivatives are in.
    let k = 10f64.ln() / SCALE;

    for _ in 0..500 {
        let mut moved: f64 = 0.0;
        for i in 0..n {
            let mut residual = 0.0; // observed minus expected score for entrant i
            let mut information = 0.0; // sum of n * E * (1 - E)

            for p in pairings {
                let (opponent, games, score) = if p.a == i {
                    (p.b, p.games, p.score_a)
                } else if p.b == i {
                    (p.a, p.games, p.games - p.score_a)
                } else {
                    continue;
                };
                if games <= 0.0 {
                    continue;
                }
                let e = expected_score(ratings[i], ratings[opponent]);
                residual += score - games * e;
                information += games * e * (1.0 - e);
            }

            // The prior: PRIOR_GAMES virtual games, half won, against the anchor rating.
            let e0 = expected_score(ratings[i], ratings[anchor]);
            residual += 0.5 * PRIOR_GAMES - PRIOR_GAMES * e0;
            information += PRIOR_GAMES * e0 * (1.0 - e0);

            if information <= 0.0 {
                continue;
            }
            let step = residual / (k * information);
            ratings[i] += step;
            moved = moved.max(step.abs());
        }

        // Re-anchor every sweep so the free additive constant cannot wander.
        let shift = ratings[anchor];
        for r in ratings.iter_mut() {
            *r -= shift;
        }

        if moved < 0.1 {
            break;
        }
    }

    let mut stderr = vec![f64::INFINITY; n];
    for i in 0..n {
        let mut information = 0.0;
        for p in pairings {
            let opponent = if p.a == i {
                p.b
            } else if p.b == i {
                p.a
            } else {
                continue;
            };
            let e = expected_score(ratings[i], ratings[opponent]);
            information += p.games * e * (1.0 - e);
        }
        let e0 = expected_score(ratings[i], ratings[anchor]);
        information += PRIOR_GAMES * e0 * (1.0 - e0);
        stderr[i] = if information > 0.0 {
            1.0 / (k * information.sqrt())
        } else {
            f64::INFINITY
        };
    }
    // The anchor is fixed by definition, not estimated.
    stderr[anchor] = 0.0;

    EloTable {
        names,
        ratings,
        stderr,
        anchor,
    }
}

impl EloTable {
    /// Entrant indices ordered strongest first.
    pub fn ranking(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.names.len()).collect();
        order.sort_by(|&a, &b| {
            self.ratings[b]
                .partial_cmp(&self.ratings[a])
                .expect("ratings are never NaN")
        });
        order
    }

    /// Expected score for entrant `a` against entrant `b` under the fitted ratings.
    pub fn expected_score(&self, a: usize, b: usize) -> f64 {
        expected_score(self.ratings[a], self.ratings[b])
    }

    /// A Markdown table, strongest first, for pasting into `FINDINGS.md`.
    pub fn markdown(&self) -> String {
        let mut out = String::from("| agent | Elo | ± | vs. anchor |\n|---|---:|---:|---:|\n");
        for i in self.ranking() {
            out.push_str(&format!(
                "| {} | {:+.0} | {:.0} | {:.3} |\n",
                self.names[i],
                self.ratings[i],
                self.stderr[i],
                self.expected_score(i, self.anchor),
            ));
        }
        out
    }
}

impl fmt::Display for EloTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "  {:<16} {:>8}  {:>6}   {}",
            "agent", "Elo", "+/-", "expected vs. anchor"
        )?;
        for i in self.ranking() {
            writeln!(
                f,
                "  {:<16} {:>+8.0}  {:>6.0}   {:.3}",
                self.names[i],
                self.ratings[i],
                self.stderr[i],
                self.expected_score(i, self.anchor),
            )?;
        }
        write!(f, "  (anchor: {} = 0)", self.names[self.anchor])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two entrants who split every game must come out equal.
    #[test]
    fn an_even_record_gives_an_even_rating() {
        let table = fit(
            vec!["a".into(), "b".into()],
            &[Pairing::new(0, 1, 500, 500, 0)],
            0,
        );
        assert!((table.ratings[1] - table.ratings[0]).abs() < 1.0);
    }

    /// The textbook calibration: 75% expected score is a 191-point gap.
    #[test]
    fn a_seventy_five_percent_score_is_about_191_elo() {
        let table = fit(
            vec!["weak".into(), "strong".into()],
            &[Pairing::new(0, 1, 2500, 7500, 0)],
            0,
        );
        assert!(
            (table.ratings[1] - 191.0).abs() < 5.0,
            "expected ~+191, got {:.1}",
            table.ratings[1]
        );
    }

    /// A perfect record is finite because of the prior, and its error bar says so.
    #[test]
    fn a_clean_sweep_stays_finite() {
        let table = fit(
            vec!["loser".into(), "winner".into()],
            &[Pairing::new(0, 1, 0, 200, 0)],
            0,
        );
        assert!(table.ratings[1].is_finite());
        assert!(table.ratings[1] > 400.0);
        assert!(
            table.stderr[1] > 100.0,
            "a rating held up only by the prior must carry a wide error bar, got {:.0}",
            table.stderr[1]
        );
    }

    /// Transitivity: the fit reconstructs a consistent ordering from indirect evidence.
    #[test]
    fn ratings_recover_a_transitive_ordering() {
        let names = vec!["c".into(), "b".into(), "a".into()];
        let pairings = [
            Pairing::new(0, 1, 3000, 7000, 0), // c loses to b
            Pairing::new(1, 2, 3000, 7000, 0), // b loses to a
        ];
        let table = fit(names, &pairings, 0);
        assert!(table.ratings[2] > table.ratings[1]);
        assert!(table.ratings[1] > table.ratings[0]);
        assert_eq!(table.ranking(), vec![2, 1, 0]);
        // c vs a was never played; the model still predicts it. Two 70% steps are ~147 Elo
        // each, so the composed gap of ~294 predicts about 0.845.
        assert!(
            table.expected_score(2, 0) > 0.82,
            "indirect prediction came out at {:.3}",
            table.expected_score(2, 0)
        );
    }

    /// Fitting is a batch operation, so the order the pairings arrive in cannot matter.
    #[test]
    fn the_fit_is_independent_of_pairing_order() {
        let names: Vec<String> = vec!["x".into(), "y".into(), "z".into()];
        let forward = [
            Pairing::new(0, 1, 600, 400, 0),
            Pairing::new(1, 2, 700, 300, 0),
            Pairing::new(0, 2, 800, 200, 0),
        ];
        let mut backward = forward;
        backward.reverse();
        let a = fit(names.clone(), &forward, 0);
        let b = fit(names, &backward, 0);
        for i in 0..3 {
            assert!((a.ratings[i] - b.ratings[i]).abs() < 0.5);
        }
    }
}
