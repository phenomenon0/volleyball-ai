import streamlit as st
import pandas as pd
import random
from anthropic import Anthropic
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

TEAMS = {
    "Penn State Nittany Lions": "penn_state",
    "Nebraska Huskers": "nebraska",
    "Texas Longhorns": "texas",
    "Wisconsin Badgers": "wisconsin",
    "Ohio St. Buckeyes": "ohio_state",
    "Minnesota Golden Gophers": "minnesota",
    "Purdue Boilermakers": "purdue",
    "Oregon Ducks": "oregon",
}


@st.cache_data
def load_team_data(prefix):
    team_stats = pd.read_csv(DATA_DIR / f"{prefix}_team_stats.csv", skiprows=2)
    off = pd.read_csv(DATA_DIR / f"{prefix}_player_offense.csv")
    defs = pd.read_csv(DATA_DIR / f"{prefix}_player_defense.csv")
    meta = pd.read_csv(DATA_DIR / f"{prefix}_team_stats.csv", nrows=1)
    # Merge player offense + defense
    players = off.merge(defs, on=["number", "player"], how="outer", suffixes=("", "_def"))
    return meta, team_stats, players


def get_team_meta(prefix):
    meta = pd.read_csv(DATA_DIR / f"{prefix}_team_stats.csv", nrows=1)
    return meta.iloc[0]


def build_team_context(meta, team_stats, players):
    lines = [f"Team: {meta['team']} | Record: {meta['record']} | Conf: {meta['conf_record']}"]

    # Filter duplicate per-set stats that just repeat existing rows
    skip_stats = {"Kills Per Set", "Assists Per Set", "Dig Per Set", "Blocks Per Set"}
    lines.append("\nTeam Stats (per set, team vs opponent):")
    for _, r in team_stats.iterrows():
        if r["stat"] in skip_stats:
            continue
        lines.append(f"  {r['stat_category']} - {r['stat']}: {r['team_per_set']} vs {r['opponent_per_set']}")

    # Pre-compute key differentials for the model
    lines.append("\nKey Differentials (team - opponent):")
    for _, r in team_stats.iterrows():
        if r["stat"] in skip_stats:
            continue
        team_val = pd.to_numeric(r['team_per_set'], errors='coerce')
        opp_val = pd.to_numeric(r['opponent_per_set'], errors='coerce')
        if pd.notna(team_val) and pd.notna(opp_val) and opp_val != 0:
            raw_diff = round(team_val - opp_val, 3)
            diff_pct = round((team_val - opp_val) / abs(opp_val) * 100, 1)
            lines.append(f"  {r['stat']}: {raw_diff:+} ({diff_pct:+.1f}%)")

    # Compute scoring breakdown from team totals row
    team_row = players[(players["number"] == "-") & (players["player"] != "Opponent")]
    if not team_row.empty:
        t = team_row.iloc[0]
        t_pts = pd.to_numeric(t.get("pts", 0), errors="coerce") or 1
        t_k = pd.to_numeric(t.get("k", 0), errors="coerce") or 0
        t_sa = pd.to_numeric(t.get("sa", 0), errors="coerce") or 0
        t_blk = pd.to_numeric(t.get("blk", 0), errors="coerce") or 0
        t_se = pd.to_numeric(t.get("se", 0), errors="coerce") or 0
        k_pct = round(t_k / t_pts * 100, 1)
        sa_pct = round(t_sa / t_pts * 100, 1)
        blk_pct = round(t_blk / t_pts * 100, 1)
        sa_err_ratio = f"{t_sa:.1f}:{t_se:.1f}"
        lines.append(f"\nScoring Breakdown: {k_pct}% kills, {sa_pct}% aces, {blk_pct}% blocks")
        lines.append(f"Team Serve Ace:Error Ratio: {sa_err_ratio}")

    # Opponent serve ace:error ratio
    opp_row = players[players["player"] == "Opponent"]
    if not opp_row.empty:
        o = opp_row.iloc[0]
        o_sa = pd.to_numeric(o.get("sa", 0), errors="coerce") or 0
        o_se = pd.to_numeric(o.get("se", 0), errors="coerce") or 0
        lines.append(f"Opponent Serve Ace:Error Ratio: {o_sa:.1f}:{o_se:.1f}")

    # Top offensive contributors by volume and efficiency
    real = players[(players["number"] != "-") & (players["player"] != "Opponent")].copy()
    real["pts_num"] = pd.to_numeric(real["pts"], errors="coerce").fillna(0)
    real["k_num"] = pd.to_numeric(real["k"], errors="coerce").fillna(0)
    real["pct_num"] = pd.to_numeric(real["pct"], errors="coerce").fillna(0)
    real["dig_num"] = pd.to_numeric(real.get("dig", pd.Series(dtype=float)), errors="coerce").fillna(0)
    real["blk_num"] = pd.to_numeric(real.get("blk", pd.Series(dtype=float)), errors="coerce").fillna(0)
    real["a_num"] = pd.to_numeric(real["a"], errors="coerce").fillna(0)

    lines.append("\nTop Offensive Contributors (by volume — pts/set):")
    for _, r in real.nlargest(5, "pts_num").iterrows():
        lines.append(f"  #{r['number']} {r['player']}: {r['pts']} pts/set, {r['k']} K/set, {r['pct']} hit%")

    lines.append("\nTop Efficiency (hit% among players with ≥1 K/set):")
    eff = real[real["k_num"] >= 1].nlargest(5, "pct_num")
    for _, r in eff.iterrows():
        lines.append(f"  #{r['number']} {r['player']}: {r['pct']} hit%, {r['k']} K/set")

    lines.append("\nTop Defensive Contributors:")
    for _, r in real.nlargest(3, "dig_num").iterrows():
        lines.append(f"  #{r['number']} {r['player']}: {r.get('dig','?')} DIG/set, {r.get('blk','?')} BLK/set")

    lines.append("\nTop Blockers:")
    for _, r in real.nlargest(3, "blk_num").iterrows():
        lines.append(f"  #{r['number']} {r['player']}: {r.get('blk','?')} BLK/set, {r.get('bs','?')} BS, {r.get('ba','?')} BA")

    # Setter identification
    setter = real.nlargest(1, "a_num")
    if not setter.empty:
        s = setter.iloc[0]
        lines.append(f"\nPrimary Setter: #{s['number']} {s['player']} — {s['a']} A/set")

    return "\n".join(lines)


def build_player_context(row, players, meta):
    lines = [f"#{row['number']} {row['player']}"]
    # Offense
    lines.append(f"Offense: {row.get('pts',0)} pts/set, {row.get('k',0)} K/set, {row.get('pct',0)} hit%, {row.get('a',0)} A/set, {row.get('sa',0)} SA/set, {row.get('se',0)} SE/set, {row.get('ae',0)} AE/set, {row.get('ta',0)} TA/set")
    # Defense
    lines.append(f"Defense: {row.get('dig',0)} DIG/set, {row.get('blk',0)} BLK/set, {row.get('bs',0)} BS, {row.get('ba',0)} BA, {row.get('re',0)} RE/set, {row.get('ra',0)} RA/set")

    # Serve ace:error ratio
    sa_num = pd.to_numeric(row.get('sa', 0), errors="coerce") or 0
    se_num = pd.to_numeric(row.get('se', 0), errors="coerce") or 0
    lines.append(f"Serve Ace:Error Ratio: {sa_num:.1f}:{se_num:.1f}")

    # Team totals for context (so model can compute contribution %)
    team_row = players[players["number"] == "-"]
    t = team_row.iloc[0] if not team_row.empty else None
    if t is not None:
        lines.append(f"\nTeam Totals: {t.get('pts',0)} pts/set, {t.get('k',0)} K/set, {t.get('a',0)} A/set, {t.get('dig','?')} DIG/set, {t.get('blk','?')} BLK/set")
        t_pts = pd.to_numeric(t.get("pts", 0), errors="coerce") or 1
        player_pts = pd.to_numeric(row.get('pts', 0), errors='coerce') or 0
        pts_pct = round(player_pts / t_pts * 100, 1) if t_pts > 0 else 0
        lines.append(f"Share of Team Scoring: {pts_pct}%")

    # Positional inference hints
    k_num = pd.to_numeric(row.get('k', 0), errors="coerce") or 0
    a_num = pd.to_numeric(row.get('a', 0), errors="coerce") or 0
    dig_num = pd.to_numeric(row.get('dig', 0), errors="coerce") or 0
    blk_num = pd.to_numeric(row.get('blk', 0), errors="coerce") or 0
    re_num = pd.to_numeric(row.get('re', 0), errors="coerce") or 0
    ra_num = pd.to_numeric(row.get('ra', 0), errors="coerce") or 0

    # Hitting efficiency vs team average (for players with >=1 K/set)
    pct_val = pd.to_numeric(row.get('pct', 0), errors="coerce")
    team_pct = pd.to_numeric(t.get('pct', 0), errors="coerce") if t is not None else None
    if k_num >= 1.0 and pct_val is not None and team_pct is not None and pct_val == pct_val and team_pct == team_pct:
        diff_pts = round((pct_val - team_pct) * 1000)
        direction = "above" if diff_pts >= 0 else "below"
        lines.append(f"Hitting Efficiency vs Team: {abs(diff_pts)} points {direction} team {team_pct}")

    # Receiver check: has reception attempts
    is_receiver = ra_num > 0.1
    lines.append(f"\nReceiver (passes serve receive): {'Yes' if is_receiver else 'No'}")

    # Infer position category with evaluation lens
    if a_num >= 5:
        pos_hint = "Setter — evaluate distribution, tempo-setting, and defensive involvement"
    elif blk_num >= 0.8 and dig_num < 1.0:
        pos_hint = "Middle Blocker — evaluate blocking anchor role, quick-tempo efficiency, and slide game"
    elif k_num >= 2.0 and is_receiver:
        pos_hint = "Outside Hitter — evaluate pin attacking, serve-receive, and all-around game"
    elif k_num >= 1.5 and not is_receiver:
        pos_hint = "Opposite/Right-side Hitter — evaluate scoring volume, out-of-system attacking, and blocking"
    elif dig_num >= 2.0 and k_num < 0.5:
        pos_hint = "Libero/DS — evaluate passing platform, defensive range, and serve-receive reliability"
    else:
        pos_hint = "Role Player/Utility — evaluate specific-rotation contributions and specialist value"
    lines.append(f"Likely Position: {pos_hint}")

    return "\n".join(lines)


MIN_PTS_PER_SET = 0.3  # filter out low-volume bench players


def filter_real_players(players):
    """Return only players with meaningful playing time."""
    real = players[(players["number"] != "-") & (players["player"] != "Opponent")].copy()
    real["pts_num"] = pd.to_numeric(real["pts"], errors="coerce").fillna(0)
    return real[real["pts_num"] >= MIN_PTS_PER_SET]


def call_claude(prompt):
    client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# --- UI ---
st.set_page_config(page_title="Volleyball AI Scout", layout="centered")


def check_password():
    """Simple password gate."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if st.session_state["authenticated"]:
        return True
    st.title("Volleyball AI Scout")
    pwd = st.text_input("Password", type="password")
    if pwd == "volleyball":
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd:
        st.error("Incorrect password")
    return False


if not check_password():
    st.stop()

st.title("Volleyball AI Scout")

team_names = list(TEAMS.keys())

# Random buttons
col_rand1, col_rand2, col_spacer = st.columns([1, 1, 2])
with col_rand1:
    if st.button("Random Team"):
        st.session_state["rand_team"] = random.choice(team_names)
        st.session_state["rand_report"] = "Team"
        st.session_state["auto_generate"] = True
with col_rand2:
    if st.button("Random Player"):
        rand_team = random.choice(team_names)
        st.session_state["rand_team"] = rand_team
        st.session_state["rand_report"] = "Player"
        st.session_state["auto_generate"] = True

# Apply random selections if set
default_team = st.session_state.pop("rand_team", team_names[0])
default_report = st.session_state.pop("rand_report", "Team")
auto_generate = st.session_state.pop("auto_generate", False)

team_idx = team_names.index(default_team) if default_team in team_names else 0
team_name = st.selectbox("Select Team", team_names, index=team_idx)
prefix = TEAMS[team_name]
meta, team_stats, players = load_team_data(prefix)
meta_row = meta.iloc[0]

report_type = st.radio("Report Type", ["Team", "Player"], horizontal=True,
                       index=0 if default_report == "Team" else 1)

if report_type == "Team":
    prompt_style = st.radio("Prompt Style", ["Narrative", "Scouting Card"], horizontal=True)

if report_type == "Player":
    filtered = filter_real_players(players)
    player_names = filtered["player"].tolist()
    # Pick a random player if triggered by Random Player button
    if default_report == "Player" and auto_generate and player_names:
        rand_player = random.choice(player_names)
        player_idx = player_names.index(rand_player)
    else:
        player_idx = 0
    player_name = st.selectbox("Select Player", player_names, index=player_idx)
    player_row = filtered[filtered["player"] == player_name].iloc[0]

if st.button("Generate Report", type="primary") or auto_generate:
    with st.spinner("Generating..."):
        if report_type == "Team":
            context = build_team_context(meta_row, team_stats, players)
            if prompt_style == "Narrative":
                prompt = f"""Volleyball team intelligence report. STRICT format rules:

- Start with a 2-sentence profile packed with the top-line numbers a coach needs at a glance: team name, system, record, hitting % vs opponent hitting %, hitting differential, kills/set vs opponent, and blocks/set vs opponent. No adjectives — just stats and what they mean. Example: "**Nebraska** runs a **5-1** at **33-1**, hitting **.351** vs opponents' **.129** — a **+.222** differential. The offense produces **14.7 kills/set** vs **10.2** allowed and outblocks opponents **2.7** to **1.5/set**."
- Then two sections: **Offense** and **Defense**
- 4 bullets per section. ONE sentence per bullet, MAX 30 words. No multi-sentence bullets.
- **Bold** every stat inline
- NEVER start a bullet with a stat label like "Digs", "Blocks", "Serving". Start with the team name or a verb.
- No player names. No filler adjectives — say what IS with numbers.

Example bullets (match this density):
- Scoring breaks **60% kills, 5% aces, 11% blocks** — a kill-driven attack that scores at the net, not from the line
- The back row digs **13.6/set** to opponents' **10.1** — a **+35%** differential feeding transition
- Serve-receive holds at **.934** with **1.0 RE/set**, nearly matching opponents' **.940** — clean but not a separator

**Offense** covers (pick 4): scoring breakdown, distribution/tempo, serving pressure, service %, kill differential
**Defense** covers (pick 4): opponent suppression, blocking, digging, serve-receive/errors, one vulnerability

{context}"""
            else:
                prompt = f"""Volleyball team scouting card. STRICT format rules:

- First line: team name, record, conference record as a header. Example: "**Nebraska Huskers — 33-1 | Conf: 20-0**"
- Then an **Identity:** line — ONE sentence describing this team's character based on what the stats reveal. No stats in this line, just a plain-language description of who they are and how they play. Example: "**Identity:** Elite, suffocating team that dominates on both sides of the net with high-efficiency attacking and relentless defense."
- Then 6 bullets covering: Attack, Serve, Reception, Defense, Blocking, Exploit
- Each bullet starts with a **bold category label** followed by a colon, then stats with a dash and a short interpretation woven together
- **Bold** every stat inline
- The Exploit bullet identifies the one data-backed vulnerability an opponent should target
- Keep each bullet to ONE sentence, dense with stats and meaning unified — not stats then meaning, but stats AS meaning in one flow

Example bullets (match this format exactly):
- **Attack:** **14.7 K/set** vs **10.2** allowed; **.351 hit%** vs **.129** opp — massive efficiency gap, best-in-class offense
- **Serve:** **1.3 aces/set** vs **1.0** opp; **91.8% serve%** — controlled aggression, few free points given
- **Reception:** **0.934 rec%** with only **1.0 errors/set** vs **1.3** opp — clean passing fuels their offense
- **Defense:** **13.6 digs/set** vs **10.1** opp — elite floor defense extends rallies consistently
- **Blocking:** **2.7 blocks/set** vs **1.6** opp — dominant front wall funnels hitters into diggers
- **Exploit:** Reception percentage is near-identical (**.934** vs **.940**) — aggressive serving is their one crackable seam; disrupt first-ball passing to prevent their efficient swing

{context}"""
        else:
            context = build_player_context(player_row, players, meta_row)
            team_ctx = f"Team: {meta_row['team']} ({meta_row['record']})"
            pts_per_set = pd.to_numeric(player_row.get('pts', 0), errors="coerce") or 0
            team_total_row = players[players["number"] == "-"]
            team_pts = pd.to_numeric(team_total_row.iloc[0].get('pts', 1), errors="coerce") if not team_total_row.empty else 1
            pts_pct = round((pts_per_set / team_pts) * 100, 1) if team_pts > 0 else 0
            is_low_usage = pts_per_set < 0.5
            bullet_count = "2-3" if is_low_usage else "4"
            usage_note = f"\nThis player contributes {pts_pct}% at {pts_per_set} pts/set — reserve/specialist. Write 3-4 bullets max. First bullet MUST note the limited sample. Do not apply evaluative labels to tiny samples." if is_low_usage else ""

            prompt = f"""Volleyball player intelligence report. STRICT format rules:

- Start with a 2-sentence profile: name, position, team, scoring share, and a character sketch that covers their role + primary value. Example: "**Harper Murray** — six-rotation OH for Nebraska (**33-1**), carrying **16.3%** of team scoring at **4.0 pts/set**. The go-to pin who absorbs the highest volume on the roster at the cost of efficiency, hitting **.295** against the team's **.351**."
- Then 4 bullets covering the most important remaining observations — no section headers, no sub-bullets
- ONE sentence per bullet, MAX 30 words. No multi-sentence bullets.
- **Bold** every stat inline
- NEVER start a bullet with a stat label like "Digs", "Blocks", "Hitting", "Serving". Start with the player's name, "She/He", or a verb describing what they do.
- Player-vs-team comparisons for context
- Real volleyball language. No filler adjectives.

Example bullets (match this length, density, and sentence starts exactly):
- Murray carries **16.3%** of Nebraska's scoring at **4.0 pts/set**, the highest-volume attacker on a **33-1** squad
- She hits **.295** on **8.4 attempts/set** — **56 points below** team **.351**, lowest efficiency among starters
- Her serve game runs **0.3 aces** against **0.4 errors** for a **0.75:1 ratio** — conservative, prioritizing in-system play

Position bullet order: OH — role/share, efficiency vs team, kill volume, serve-receive, serving, defense, exposure. MB — role/share, efficiency, blocking, kill volume, serving, exposure. Setter — role/system, distribution, blocking, digs, scoring, serving, exposure. Libero — role, passing, digging, serving, exposure.
Skip zero/irrelevant stats. Last bullet is always the exposure — one data-backed weakness to target.
{usage_note}

{team_ctx}
{context}"""

        report = call_claude(prompt)
        st.markdown("---")
        st.markdown(report)

        # Raw stats for fact-checking
        with st.expander("Raw Stats (fact-check)"):
            if report_type == "Team":
                st.caption(f"{meta_row['team']} | Record: {meta_row['record']} | Conf: {meta_row['conf_record']}")
                st.dataframe(team_stats, use_container_width=True, hide_index=True)
            else:
                st.caption(f"#{player_row['number']} {player_row['player']} — {meta_row['team']} ({meta_row['record']})")
                # Show player stats as a clean table
                off_cols = ['pts', 'k', 'k_per_set', 'ae', 'ta', 'pct', 'a', 'a_per_set', 'sa', 'sa_per_set', 'se']
                def_cols = ['dig', 'dig_per_set', 're', 're_per_set', 'bs', 'ba', 'blk', 'blk_per_set']
                available_off = [c for c in off_cols if c in player_row.index]
                available_def = [c for c in def_cols if c in player_row.index]
                if available_off:
                    st.markdown("**Offense**")
                    st.dataframe(player_row[available_off].to_frame().T, use_container_width=True, hide_index=True)
                if available_def:
                    st.markdown("**Defense**")
                    st.dataframe(player_row[available_def].to_frame().T, use_container_width=True, hide_index=True)
