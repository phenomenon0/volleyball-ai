import streamlit as st
import pandas as pd
from anthropic import AnthropicBedrock
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

TEAMS = {
    "Penn State Nittany Lions": "penn_state",
    "Nebraska Huskers": "nebraska",
    "Texas Longhorns": "texas",
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
    lines.append("\nTeam Stats (per set, team vs opponent):")
    for _, r in team_stats.iterrows():
        lines.append(f"  {r['stat_category']} - {r['stat']}: {r['team_per_set']} vs {r['opponent_per_set']}")
    return "\n".join(lines)


def build_player_context(row):
    lines = [f"#{row['number']} {row['player']}"]
    # Offense
    lines.append(f"Offense: {row.get('k',0)} K/set, {row.get('pct',0)} hit%, {row.get('a',0)} A/set, {row.get('sa',0)} SA/set, {row.get('se',0)} SE/set, {row.get('ae',0)} AE/set")
    # Defense
    lines.append(f"Defense: {row.get('dig',0)} DIG/set, {row.get('blk',0)} BLK/set, {row.get('bs',0)} BS, {row.get('ba',0)} BA, {row.get('re',0)} RE/set")
    return "\n".join(lines)


def call_bedrock(prompt):
    client = AnthropicBedrock(
        aws_profile="SandboxAdmin_Sandbox_1355-639769761355",
        aws_region="us-east-1",
    )
    resp = client.messages.create(
        model="us.anthropic.claude-opus-4-6-v1",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# --- UI ---
st.set_page_config(page_title="Volleyball AI Scout", layout="centered")
st.title("Volleyball AI Scout")

team_name = st.selectbox("Select Team", list(TEAMS.keys()))
prefix = TEAMS[team_name]
meta, team_stats, players = load_team_data(prefix)
meta_row = meta.iloc[0]

report_type = st.radio("Report Type", ["Team", "Player"], horizontal=True)

if report_type == "Player":
    real_players = players[players["number"] != "-"]
    player_name = st.selectbox("Select Player", real_players["player"].tolist())
    player_row = real_players[real_players["player"] == player_name].iloc[0]

if st.button("Generate Report", type="primary"):
    with st.spinner("Generating..."):
        if report_type == "Team":
            context = build_team_context(meta_row, team_stats, players)
            prompt = f"""You are a volleyball scout writing a quick-skim bullet report for coaches. Given this team data, produce structured markdown EXACTLY in this format:

# {{Team Name}} — {{Record}} | Conf: {{Conf Record}}

**Identity:** One sentence team profile (style of play, tempo, identity).

- **Attack:** Bold the key stat (e.g. **13.5 K/set**), compare team vs opponent, one-line takeaway
- **Serve:** Bold the key stat, compare team vs opponent, one-line takeaway
- **Reception:** Bold the key stat, compare team vs opponent, one-line takeaway
- **Defense:** Bold the key stat (digs), compare team vs opponent, one-line takeaway
- **Blocking:** Bold the key stat, compare team vs opponent, one-line takeaway
- **Exploit:** One tactical takeaway for an opposing coach — the biggest vulnerability to attack

Rules: Max ~100 words total. Every bullet must contain a bold number. Use the team vs opponent per-set columns for comparisons. Be direct and specific.

{context}"""
        else:
            context = build_player_context(player_row)
            team_ctx = f"Team: {meta_row['team']} ({meta_row['record']})"
            pts_per_set = player_row.get('pts', 0)
            low_usage_flag = "\nIMPORTANT: This player has pts/set < 0.5. Lead the report with: ⚠️ Limited sample — reserve role" if pts_per_set < 0.5 else ""
            prompt = f"""You are a volleyball scout writing a quick-skim bullet report for coaches. Given this player data, produce structured markdown EXACTLY in this format:

# #{{Number}} {{Name}} — {{Role}} | {{Team}} ({{Record}})

**Role:** One sentence — primary function (attacker, setter, libero, DS, middle, etc.) inferred from their stats.

- **Attacking:** Bold the key stat (e.g. **5.4 K/set, .319 hit%**), one-line takeaway
- **Serving:** Bold the key stat, one-line takeaway
- **Passing/Reception:** Bold the key stat, one-line takeaway
- **Defense/Blocking:** Bold the key stat, one-line takeaway
- **Watch:** Key weakness or exploitable tendency — this bullet is always last

Rules: Max ~80 words total, 5-6 bullets. Every bullet must contain a bold number. Infer the player's role from which stats are strongest. Be direct and specific. Skip skill areas where the player has zero or negligible stats.{low_usage_flag}

{team_ctx}
{context}"""

        report = call_bedrock(prompt)
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
