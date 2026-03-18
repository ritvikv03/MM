/**
 * frontend/lib/ncaa-filter.ts
 * News feed gatekeeper — filters incoming news items to only those relevant
 * to currently active NCAA college basketball players.
 *
 * Cross-references against:
 * 1. The active tournament team roster (from /api/intel or static team names)
 * 2. A known set of NBA-departed players (fetched from ncaa_rosters.json or
 *    the backend /api/intel endpoint which includes nba_draft_class)
 *
 * A news item is EXCLUDED if it names a player who has left for the NBA.
 * A news item PASSES if it mentions an active college player or tournament team.
 */

export interface NewsItem {
  id: string;
  title: string;
  body?: string;
  team?: string;
  player?: string;
  source?: string;
  publishedAt?: string;
  url?: string;
}

export interface RosterContext {
  activeTeams: string[];           // current tournament teams
  activePlayers: string[];         // active D-I player names
  nbaDepartedPlayers: string[];    // players who left for NBA (filter these out)
}

/**
 * Build a RosterContext from the intel API response.
 * The intel response includes team names and nba_draft_class from the backend.
 */
export function buildRosterContext(
  teams: string[],
  nbaDeparted: string[] = [],
  activePlayers: string[] = [],
): RosterContext {
  return {
    activeTeams: teams,
    activePlayers,
    nbaDepartedPlayers: nbaDeparted,
  };
}

/**
 * filterNewsItems — the gatekeeper.
 *
 * Cross-references each news item against RosterContext:
 * - Excludes items mentioning NBA-departed players by name
 * - Passes items mentioning active teams or players
 * - Excludes items with no college basketball relevance
 *
 * Matching is case-insensitive and checks both title and body.
 */
export function filterNewsItems(
  items: NewsItem[],
  context: RosterContext,
): NewsItem[] {
  if (!items.length) return [];

  const nbaDepartedLower = new Set(
    context.nbaDepartedPlayers.map(p => p.toLowerCase())
  );
  const activeTeamsLower = new Set(
    context.activeTeams.map(t => t.toLowerCase())
  );
  const activePlayersLower = new Set(
    context.activePlayers.map(p => p.toLowerCase())
  );

  return items.filter(item => {
    const combined = `${item.title} ${item.body ?? ''} ${item.player ?? ''}`.toLowerCase();

    // EXCLUDE: mentions NBA-departed player
    for (const nbaPlayer of nbaDepartedLower) {
      if (combined.includes(nbaPlayer)) return false;
    }

    // PASS: mentions active tournament team
    for (const team of activeTeamsLower) {
      if (combined.includes(team)) return true;
    }

    // PASS: item's team field is an active team
    if (item.team && activeTeamsLower.has(item.team.toLowerCase())) return true;

    // PASS: mentions active player by name
    for (const player of activePlayersLower) {
      if (combined.includes(player)) return true;
    }

    // EXCLUDE: no college basketball relevance found
    return false;
  });
}

/**
 * isPlayerActive — point lookup to check if a specific player name
 * is an active college player (not NBA-departed).
 *
 * Used in inline rendering logic to suppress individual player references.
 */
export function isPlayerActive(
  playerName: string,
  context: RosterContext,
): boolean {
  const nameLower = playerName.toLowerCase();
  const isDeparted = context.nbaDepartedPlayers.some(
    p => p.toLowerCase() === nameLower
  );
  return !isDeparted;
}

/**
 * getTeamNewsFilter — returns a filter function pre-bound to a specific team.
 * Useful for the War Room and Projections intel feeds.
 */
export function getTeamNewsFilter(
  teamName: string,
  context: RosterContext,
): (items: NewsItem[]) => NewsItem[] {
  const teamLower = teamName.toLowerCase();
  return (items: NewsItem[]) =>
    filterNewsItems(
      items.filter(item => {
        const combined = `${item.title} ${item.body ?? ''} ${item.team ?? ''}`.toLowerCase();
        return combined.includes(teamLower);
      }),
      context,
    );
}
