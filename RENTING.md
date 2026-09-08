# Renting cores for the 12-hour run

The operational half of `PLAN.md` item 7. `configs/train-12h.toml` says what to run; this says
how to get a machine to run it on, for someone who has never rented one.

Read the two boxes below first. Everything else is typing.

> **You pay from the moment the server exists to the moment you DELETE it** — not while it is
> computing, and on Hetzner **not** only while it is powered on. A stopped server still bills,
> because the resources stay reserved for you. The only thing that stops the meter is deletion.
> Set a timer on your phone for 13 hours when you create it.

> **A dropped SSH connection kills whatever it was running.** Closing the laptop lid, losing
> wifi, or a hotel captive portal will end a 12-hour run at hour 3 with nothing written. The
> fix is one command — `tmux` — and it is step 8. Do not skip it.

## 0. Which provider

**Hetzner Cloud**, for this run, unless you already have an AWS account and habits.

| | Hetzner Cloud | AWS EC2 |
|---|---|---|
| 12 hours, ~48–64 cores | **~€5** | ~\$40 on-demand, ~\$15 spot |
| Setup before you can launch | account + SSH key | account + VPC + security group + key pair + AMI + quota request |
| Can it be taken away mid-run | no | yes, if spot |
| Billing granularity | hourly | per second |

The job is a 12-hour CPU batch with no network dependency, no service to expose, and no data
that has to live anywhere in particular. That is the shape Hetzner is cheapest and simplest at,
and the €35 difference is not the point — the point is that AWS has five more places to make a
first-time mistake, and each of them costs an evening rather than a euro.

⚠️ **Sign up a day or two before you want to run.** New Hetzner accounts get identity/payment
verification (minutes to a day), and they start with a resource limit that may not let you
create a 48-vCPU server immediately. Both are cleared by asking support, usually same day, but
not at 9pm on the night you wanted to start.

If you would rather use AWS anyway, section 12 has the differences.

## 1. Your SSH key

You already have one — `~/.ssh/id_ed25519.pub`, the one that talks to GitHub. That is the file
whose *contents* you paste into the provider's web console. Print it:

```bash
cat ~/.ssh/id_ed25519.pub
```

One line beginning `ssh-ed25519`. That is the public half and it is safe to paste anywhere. The
file without `.pub` is the private half and never leaves your laptop.

If you ever need a fresh one: `ssh-keygen -t ed25519 -C "duel52"` and press enter three times.

## 2. Account and project

1. <https://console.hetzner.cloud> → sign up, verify email, add a payment method.
2. Verification may ask for ID. Do this early (see the warning above).
3. Create a **project** — call it `duel52`. A project is just a folder for servers.
4. In the project sidebar: **Security → SSH keys → Add SSH key**, paste the line from step 1,
   name it `laptop`.

Adding the key here rather than at server-creation time means it is already on the list when
you get to the next step, and it will be there next time too.

## 3. Create the server

**Servers → Add Server.** Six fields:

| Field | What to pick | Why |
|---|---|---|
| Location | Any. `Ashburn, VA` or `Hillsboro, OR` if you are in the US | It is a batch job; latency is irrelevant. Pick a close one so `rsync` at the end is quick |
| Image | **Ubuntu 24.04** | Ships Python 3.12; the trainer needs `tomllib`, which is 3.11+ |
| Type | **Dedicated vCPU → CCX** line, the largest you are allowed | "Shared vCPU" means you are competing for the core with strangers, which makes every throughput measurement a lie |
| Volume, Firewall, Backups, etc. | **skip all of it** | Backups are a percentage surcharge on a machine you are deleting tomorrow |
| SSH key | tick `laptop` | If you skip this it emails you a root password instead, which is worse in every way |
| Name | `duel52-run` | |

The dedicated-vCPU line runs roughly CCX13 (2 vCPU) up to CCX63 (48 vCPU, 192 GB RAM). **Take
the biggest one available to you.** The exact hourly price is displayed in the console as you
select — read it there rather than trusting a number in this file, and multiply by 13 to get
what the run will cost.

Click **Create & Buy now**. Sixty seconds later the server list shows an IPv4 address. That
address is the machine.

## 4. First login

```bash
ssh root@<the IPv4 address>
```

Type `yes` at the fingerprint prompt. You are root on a bare Ubuntu box in a German or American
datacentre. Nothing here is precious — if you wreck it, delete it and make another one for
another €0.45.

## 5. Toolchain

```bash
apt update && apt install -y build-essential git python3-venv python3-dev tmux curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
```

Two minutes. `tmux` in that list is the thing that saves the run in step 8.

## 6. The code

The repository is public and the shipped checkpoints are tracked in it, so `models/` — including
`duel52-split-gen031.d52nn`, which the reference panel plays against every generation — arrives
with the clone. No tokens, no deploy keys, nothing to configure.

```bash
git clone https://github.com/ZGpup/Duel52.git
cd Duel52
cargo build --release
cargo test
```

**Run the tests.** They take a couple of minutes and they are the only thing that will tell you
this unfamiliar CPU reproduces the engine's results — and "same seed + same config → identical
game" is a project invariant, not a nicety. 342 passing is the answer.

`cargo test` without `--release` is deliberate and not a typo: `Cargo.toml` keeps the test
profile in debug mode at `-O1` because the engine's debug assertions are the state-invariant
checks, and `--release` would compile exactly those away.

Then Python. **The `--index-url` is not optional**: without it `pip` downloads the CUDA build of
PyTorch, which is about 2.5 GB of GPU libraries for a machine with no GPU.

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip maturin numpy pytest
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/maturin develop --release
```

Sanity check, which fails loudly if the compiled extension did not land:

```bash
.venv/bin/python -m duel52.train check --config configs/train-12h.toml
```

## 7. Stage 1 — fifteen minutes that decide the run

`configs/train-12h.toml` has four values marked `[STAGE 1]` that are measurements of *this box*,
not constants. Measuring them is the difference between 40 generations and 20.

First, copy up the one file the clone does not have — `runs/` is gitignored, and this is the
trained lane 128×3 net that gives an honest throughput number. **From your laptop**, in another
terminal:

```bash
scp runs/sixth/checkpoints/best.d52nn root@<IP>:~/Duel52/lane3-trained.d52nn
```

Back on the box, build the two random inits you need for the depth comparison:

```bash
for b in 4 6; do
  .venv/bin/python -m duel52.nn init --arch lane --encoding-slots 21 \
    --width 128 --blocks $b --value-hidden 128 --out lane$b.d52nn
done
```

Now measure. ⚠️ **`--full-search-fraction` and `--cap-sims` are not optional here**, even though
the defaults let you leave them off: without them you measure *uncapped* self-play, which is
2.9× slower, and size the whole run against a number it will never see.

```bash
for f in lane3-trained lane4 lane6; do
  echo "== $f"
  ./target/release/duel52 selfplay --checkpoint $f.d52nn --out /tmp/s.d52sp \
    --games 1000 --sims 256 --full-search-fraction 0.25 --cap-sims 32 \
    --encoding-slots 21 --seed 1 --quiet
done
```

Read the `games/sec` line from each. Then two corrections, both of which cost real generations
if you skip them:

- ⚠️ **`lane4` and `lane6` are random inits, and a random init plays about 25% faster than a
  trained one** — `runs/sixth` went 2.52 games/sec at generation 1 to 1.85 by generation 12, as
  games lengthen from 134 to 141 decisions. So use `lane3-trained` for the *absolute* rate and
  the two random inits only for the *ratio* between depths.
- ⚠️ **Use at least 1000 games.** Self-play is statically sharded by game with no work stealing
  (`engine/src/selfplay.rs`), so on a 48-thread box a short measurement is dominated by the
  ragged tail and reads low.

Also measure whether the extra threads are real, if the box reports hyperthreads
(`nproc` vs `lscpu | grep 'Core(s)'`):

```bash
./target/release/duel52 selfplay --checkpoint lane3-trained.d52nn --out /tmp/s.d52sp \
  --games 1000 --sims 256 --full-search-fraction 0.25 --cap-sims 32 \
  --encoding-slots 21 --seed 1 --quiet --threads 24
```

against the same command with `--threads 48`. If 48 is not meaningfully faster, pin `threads` to
the physical count.

**Then set the four values** in `configs/train-12h.toml`:

| Value | From |
|---|---|
| `net.blocks` | The deepest trunk whose measured rate still leaves ≥ 25 generations. 4 is the default; take 6 only if the box is fast |
| `selfplay.games` | `(target_generation_seconds − 600) × measured_games_per_sec`, aiming at a 16–20 minute generation. Keep it in the 6,000–12,000 band |
| `train.lr_schedule` | Boundaries at 40% and 75% of the generation count 12 hours will actually buy. Key to the **high** end of your estimate — throttling early is measured at 3× the cost of annealing late (`FINDINGS.md` F4.6) |
| `run.threads` | The measured flattening point, or 0 for all cores |

`.venv/bin/python -m duel52.train check --config configs/train-12h.toml` re-prints the plan with
your numbers in it. It costs five seconds and it is the last chance to notice something wrong.

## 8. Launch — inside tmux

```bash
tmux new -s duel52
```

The prompt comes back with a green bar at the bottom. You are now inside a session that belongs
to the *server*, not to your SSH connection. Start the run:

```bash
cd ~/Duel52
.venv/bin/python -m duel52.train run --config configs/train-12h.toml \
  --run-dir runs/seventh 2>&1 | tee runs/seventh.log
```

Press **Ctrl-b** then **d**. That detaches — the run keeps going, and you get your shell back.
Now `exit` the SSH session entirely and close the laptop. Nothing is watching it and nothing
needs to.

To look in again, from anywhere:

```bash
ssh root@<IP>
tmux attach -t duel52
```

Ctrl-b d to leave again. Or without attaching at all, straight from your laptop:

```bash
ssh root@<IP> 'tail -30 ~/Duel52/runs/seventh.log'
```

## 9. What to watch, once every few hours

Three lines in the per-generation block, in order of how much they mean:

- **`gate`** — `PROMOTED` most generations is healthy. Refusals are counted consecutively and
  five in a row ends the run.
- **`reference` third column** (`netmcts:models/duel52-split-gen031.d52nn@64`) — the progress
  chart. **Its slope, never its level**: it gives gen031 a 4:1 search handicap, and in
  `runs/sixth` it read 0.515 when the honest equal-simulation number was 0.323.
- **`held-out`** — unavailable until generation 10 by design, and expect it to drift upward and
  become unreadable in the last third anyway (`FINDINGS.md` F4.6).

If you want the honest strength number mid-run without disturbing the box, pull a checkpoint
down and score it on the laptop:

```bash
scp root@<IP>:~/Duel52/runs/seventh/checkpoints/gen020.d52nn /tmp/
./target/release/duel52 match --a netmcts:/tmp/gen020.d52nn@256 \
  --b netmcts:models/duel52-split-gen031.d52nn@256 \
  --games 200 --encoding-slots 21 --variant split --stalemate-value 0.0 --seed 1
```

## 10. Collect the results

The run stops itself when another generation would not fit the budget. **From the laptop**:

```bash
rsync -avz --exclude 'shards' root@<IP>:~/Duel52/runs/seventh/ runs/seventh/
scp root@<IP>:~/Duel52/runs/seventh.log runs/
```

Excluding `shards` skips a gigabyte or two of trajectories that are reproducible from the config
and the seed. What you want is `checkpoints/`, `log.jsonl` and `train.toml.used` — a few hundred
megabytes.

Check it arrived before you do the next step:

```bash
ls runs/seventh/checkpoints/ && wc -l runs/seventh/log.jsonl
```

## 11. Delete the server

**Servers → `duel52-run` → the ⋯ menu → Delete.** Type the name to confirm.

Then look at **Servers** and confirm the list is empty, and at any **Volumes**, **Snapshots**,
**Floating IPs** or **Load Balancers** tabs and confirm the same — those are billed separately
and survive the server that made them. You created none of them if you followed step 3, but
looking costs ten seconds and forgetting costs a monthly bill for something you cannot remember
making.

The billing page shows the accrued total. It should read a few euros.

## 12. If you use AWS instead

Same run, more ceremony. The differences that actually bite:

- **Quotas.** A fresh account cannot launch a 64-vCPU instance. The limit is expressed in vCPUs
  per instance family and raising it is a support ticket that can take a day. Check
  *Service Quotas → EC2 → Running On-Demand Standard instances* **before** you plan an evening
  around it.
- **Instance type.** `c7a.16xlarge` (64 vCPU, AMD Genoa) or `c7i.16xlarge` (Intel). Compute
  optimised, not general purpose.
- **Networking.** You must attach a security group allowing inbound TCP 22 from your IP, or you
  cannot log in. This is the step everyone misses once.
- **AMI.** Ubuntu 24.04, and the login user is `ubuntu`, not `root` — so `ssh ubuntu@<IP>` and
  `sudo` in front of the `apt` line.
- **Storage is separate and survives.** Give the root volume 60 GB, and note that terminating
  the instance deletes it only if "delete on termination" is set, which it is by default for the
  root volume and is not for anything you add.
- **Stopping ≠ terminating.** Stopping halts instance charges but keeps billing the EBS volume.
  **Terminate** when done.
- **Spot is ~⅓ the price and is safe for this run** — `--resume` works and the learning-rate
  schedule is generation-keyed precisely so that a reclaimed instance does not silently restart
  at full rate. But it can vanish with two minutes' notice, so only take it if you are willing
  to log back in and type the `--resume` command.

Everything from section 5 onward is identical.
