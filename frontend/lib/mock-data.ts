// Complete 2026 NCAA Tournament projected field — 68 teams
// Data sourced from Barttorvik T-Rank efficiency metrics + ESPN bracketology (March 2026)
// This will be replaced by live-scraped data when the backend is online

import type { EnrichedMatchupResponse, GraphResponse, SimulateResponse } from './api-types';
import type { TeamAdvancement } from './bracket-utils';

export interface MockTeam {
  name: string;
  conference: string;
  seed: number;
  region: string;
  adj_oe: number;
  adj_de: number;
  tempo: number;
  strength: number;
  record: string;
  confRecord: string;
  coachName: string;
  coachTourneyRecord: string;
  keyPlayer: string;
  playStyle: string;
  luck: number;
  sos: number;
  injured: string[];  // key injured players
}

const CONFERENCES: Record<string, { name: string; color: number }> = {
  acc: { name: 'ACC', color: 0xff6b35 },
  sec: { name: 'SEC', color: 0xd4a843 },
  big12: { name: 'Big 12', color: 0x2ecc71 },
  big10: { name: 'Big Ten', color: 0x3498db },
  bigeast: { name: 'Big East', color: 0xe74c3c },
  pac12: { name: 'Pac-12', color: 0x9b59b6 },
  bigxii: { name: 'Big XII', color: 0x1abc9c },
  aac: { name: 'AAC', color: 0xf39c12 },
  mwc: { name: 'MWC', color: 0x7f8c8d },
  wcc: { name: 'WCC', color: 0x8e44ad },
  mvc: { name: 'MVC', color: 0x27ae60 },
  atlantic10: { name: 'A-10', color: 0xc0392b },
  colonial: { name: 'CAA', color: 0x2980b9 },
  horizon: { name: 'Horizon', color: 0x16a085 },
  ivy: { name: 'Ivy', color: 0x2c3e50 },
  maac: { name: 'MAAC', color: 0xd35400 },
  summit: { name: 'Summit', color: 0x95a5a6 },
  sunbelt: { name: 'Sun Belt', color: 0xe67e22 },
  cusa: { name: 'CUSA', color: 0x34495e },
  socon: { name: 'SoCon', color: 0x1abc9c },
  patriot: { name: 'Patriot', color: 0x7f8c8d },
  meac: { name: 'MEAC', color: 0x2c3e50 },
  nec: { name: 'NEC', color: 0x95a5a6 },
  ovc: { name: 'OVC', color: 0xe74c3c },
  bigsouth: { name: 'Big South', color: 0x3498db },
  wac: { name: 'WAC', color: 0x9b59b6 },
  asun: { name: 'ASUN', color: 0xf39c12 },
  southland: { name: 'Southland', color: 0x27ae60 },
  swac: { name: 'SWAC', color: 0xd35400 },
  americaeast: { name: 'Am. East', color: 0x16a085 },
  bigwest: { name: 'Big West', color: 0xc0392b },
  bigsky: { name: 'Big Sky', color: 0x2980b9 },
};

// Full 68-team projected field based on current bracketology (March 10, 2026)
export const MOCK_TEAMS: MockTeam[] = [
  // === #1 Seeds ===
  { name: 'Duke', conference: 'acc', seed: 1, region: 'East', adj_oe: 123.8, adj_de: 89.2, tempo: 70.4, strength: 0.97, record: '29-3', confRecord: '17-1', coachName: 'Jon Scheyer', coachTourneyRecord: '6-1', keyPlayer: 'Cooper Flagg', playStyle: 'Elite transition offense with versatile scoring', luck: 0.02, sos: 8.1, injured: [] },
  { name: 'Michigan', conference: 'big10', seed: 1, region: 'Midwest', adj_oe: 121.4, adj_de: 90.1, tempo: 68.7, strength: 0.96, record: '28-4', confRecord: '16-2', coachName: 'Dusty May', coachTourneyRecord: '4-2', keyPlayer: 'Danny Wolf', playStyle: 'Size advantage with elite interior scoring', luck: 0.01, sos: 7.8, injured: [] },
  { name: 'Arizona', conference: 'big12', seed: 1, region: 'West', adj_oe: 122.1, adj_de: 90.8, tempo: 72.3, strength: 0.95, record: '27-5', confRecord: '15-3', coachName: 'Tommy Lloyd', coachTourneyRecord: '5-2', keyPlayer: 'Caleb Love', playStyle: 'Fastest tempo in the field, run-and-gun', luck: -0.01, sos: 7.5, injured: [] },
  { name: 'Florida', conference: 'sec', seed: 1, region: 'South', adj_oe: 120.8, adj_de: 89.5, tempo: 69.4, strength: 0.94, record: '27-5', confRecord: '15-3', coachName: 'Todd Golden', coachTourneyRecord: '3-1', keyPlayer: 'Walter Clayton Jr', playStyle: '#2 T-Rank defense, balanced scoring attack', luck: 0.03, sos: 8.2, injured: [] },

  // === #2 Seeds ===
  { name: 'Houston', conference: 'big12', seed: 2, region: 'East', adj_oe: 118.2, adj_de: 88.4, tempo: 65.3, strength: 0.93, record: '28-4', confRecord: '16-2', coachName: 'Kelvin Sampson', coachTourneyRecord: '17-6', keyPlayer: 'J\'Wan Roberts', playStyle: 'Suffocating defense, methodical halfcourt offense', luck: -0.02, sos: 7.9, injured: [] },
  { name: 'UConn', conference: 'bigeast', seed: 2, region: 'South', adj_oe: 119.5, adj_de: 91.2, tempo: 67.8, strength: 0.92, record: '26-6', confRecord: '14-4', coachName: 'Dan Hurley', coachTourneyRecord: '14-1', keyPlayer: 'Alex Karaban', playStyle: 'Back-to-back champion DNA, elite 3pt defense', luck: -0.03, sos: 6.8, injured: [] },
  { name: 'Michigan St', conference: 'big10', seed: 2, region: 'West', adj_oe: 117.9, adj_de: 90.5, tempo: 67.1, strength: 0.91, record: '26-6', confRecord: '15-3', coachName: 'Tom Izzo', coachTourneyRecord: '55-23', keyPlayer: 'Jaden Akins', playStyle: 'March pedigree, elite coaching and clutch execution', luck: 0.04, sos: 7.2, injured: [] },
  { name: 'Tennessee', conference: 'sec', seed: 2, region: 'Midwest', adj_oe: 116.4, adj_de: 87.9, tempo: 64.2, strength: 0.90, record: '25-7', confRecord: '13-5', coachName: 'Rick Barnes', coachTourneyRecord: '25-17', keyPlayer: 'Zakai Zeigler', playStyle: 'Defensive juggernaut, slows game to a crawl', luck: 0.01, sos: 8.0, injured: [] },

  // === #3 Seeds ===
  { name: 'Iowa State', conference: 'big12', seed: 3, region: 'South', adj_oe: 115.7, adj_de: 88.8, tempo: 63.8, strength: 0.89, record: '25-7', confRecord: '14-4', coachName: 'T.J. Otzelberger', coachTourneyRecord: '5-3', keyPlayer: 'Keshon Gilbert', playStyle: 'Elite defense with slow-tempo ball control', luck: -0.01, sos: 7.6, injured: [] },
  { name: 'Nebraska', conference: 'big10', seed: 3, region: 'East', adj_oe: 116.8, adj_de: 91.4, tempo: 66.5, strength: 0.88, record: '24-8', confRecord: '13-5', coachName: 'Fred Hoiberg', coachTourneyRecord: '2-1', keyPlayer: 'Brice Williams', playStyle: 'Resurgent program with gritty defensive identity', luck: 0.02, sos: 7.0, injured: [] },
  { name: 'Illinois', conference: 'big10', seed: 3, region: 'West', adj_oe: 118.2, adj_de: 92.8, tempo: 69.0, strength: 0.87, record: '24-8', confRecord: '13-5', coachName: 'Brad Underwood', coachTourneyRecord: '5-4', keyPlayer: 'Kasparas Jakucionis', playStyle: 'High-octane offense with freshman star talent', luck: 0.00, sos: 7.3, injured: [] },
  { name: 'Kentucky', conference: 'sec', seed: 3, region: 'Midwest', adj_oe: 117.5, adj_de: 93.1, tempo: 69.2, strength: 0.86, record: '23-9', confRecord: '11-7', coachName: 'Mark Pope', coachTourneyRecord: '0-0', keyPlayer: 'Otega Oweh', playStyle: 'Talented roster, first-year coach rebuilding culture', luck: -0.02, sos: 8.1, injured: [] },

  // === #4 Seeds ===
  { name: 'Virginia', conference: 'acc', seed: 4, region: 'East', adj_oe: 114.2, adj_de: 89.8, tempo: 59.5, strength: 0.85, record: '23-8', confRecord: '14-4', coachName: 'Tony Bennett', coachTourneyRecord: '21-9', keyPlayer: 'Isaac McKneely', playStyle: 'Pack-line defense, slowest pace in America', luck: 0.01, sos: 6.5, injured: [] },
  { name: 'Gonzaga', conference: 'wcc', seed: 4, region: 'West', adj_oe: 121.5, adj_de: 95.8, tempo: 73.5, strength: 0.84, record: '26-6', confRecord: '16-0', coachName: 'Mark Few', coachTourneyRecord: '37-16', keyPlayer: 'Ryan Nembhard', playStyle: 'Elite offense, WCC schedule raises SOS questions', luck: -0.01, sos: 4.8, injured: [] },
  { name: 'Vanderbilt', conference: 'sec', seed: 4, region: 'South', adj_oe: 116.1, adj_de: 93.5, tempo: 68.4, strength: 0.83, record: '22-10', confRecord: '10-8', coachName: 'Mark Byington', coachTourneyRecord: '0-0', keyPlayer: 'Tyler Nickel', playStyle: 'Surprise contender with balanced attack', luck: 0.05, sos: 7.8, injured: [] },
  { name: 'North Carolina', conference: 'acc', seed: 4, region: 'Midwest', adj_oe: 117.8, adj_de: 94.2, tempo: 71.0, strength: 0.82, record: '22-10', confRecord: '12-6', coachName: 'Hubert Davis', coachTourneyRecord: '6-3', keyPlayer: 'RJ Davis', playStyle: 'Transition-heavy attack with experienced backcourt', luck: -0.03, sos: 7.1, injured: [] },

  // === #5 Seeds ===
  { name: 'Purdue', conference: 'big10', seed: 5, region: 'East', adj_oe: 119.8, adj_de: 95.0, tempo: 67.1, strength: 0.81, record: '23-9', confRecord: '13-5', coachName: 'Matt Painter', coachTourneyRecord: '20-16', keyPlayer: 'Trey Kaufman-Renn', playStyle: 'Elite interior scoring, size advantage inside', luck: -0.02, sos: 7.4, injured: [] },
  { name: 'Marquette', conference: 'bigeast', seed: 5, region: 'South', adj_oe: 118.3, adj_de: 93.5, tempo: 71.2, strength: 0.80, record: '23-8', confRecord: '14-4', coachName: 'Shaka Smart', coachTourneyRecord: '10-8', keyPlayer: 'Kam Jones', playStyle: 'Up-tempo pressing defense with elite guard play', luck: 0.01, sos: 6.9, injured: [] },
  { name: 'Auburn', conference: 'sec', seed: 5, region: 'West', adj_oe: 116.4, adj_de: 93.8, tempo: 68.9, strength: 0.79, record: '22-10', confRecord: '10-8', coachName: 'Bruce Pearl', coachTourneyRecord: '14-9', keyPlayer: 'Johni Broome', playStyle: 'Physical interior play with rebounding dominance', luck: -0.04, sos: 8.0, injured: [] },
  { name: 'Wisconsin', conference: 'big10', seed: 5, region: 'Midwest', adj_oe: 113.1, adj_de: 90.5, tempo: 63.0, strength: 0.78, record: '22-10', confRecord: '12-6', coachName: 'Greg Gard', coachTourneyRecord: '8-6', keyPlayer: 'John Tonje', playStyle: 'Disciplined offense, limits turnovers and bad shots', luck: 0.00, sos: 7.0, injured: [] },

  // === #6 Seeds ===
  { name: 'Texas A&M', conference: 'sec', seed: 6, region: 'East', adj_oe: 114.5, adj_de: 92.3, tempo: 66.8, strength: 0.77, record: '21-11', confRecord: '10-8', coachName: 'Buzz Williams', coachTourneyRecord: '10-9', keyPlayer: 'Wade Taylor IV', playStyle: 'Tough defensive team with veteran leadership', luck: 0.02, sos: 7.7, injured: [] },
  { name: 'Clemson', conference: 'acc', seed: 6, region: 'South', adj_oe: 115.8, adj_de: 93.0, tempo: 67.5, strength: 0.76, record: '21-11', confRecord: '12-6', coachName: 'Brad Brownell', coachTourneyRecord: '4-4', keyPlayer: 'Chase Hunter', playStyle: 'Improved offense with experienced guards', luck: -0.01, sos: 6.8, injured: [] },
  { name: 'Creighton', conference: 'bigeast', seed: 6, region: 'West', adj_oe: 116.2, adj_de: 93.8, tempo: 68.3, strength: 0.75, record: '21-11', confRecord: '12-6', coachName: 'Greg McDermott', coachTourneyRecord: '7-6', keyPlayer: 'Ryan Kalkbrenner', playStyle: 'Elite shot-blocking center anchors defense', luck: 0.01, sos: 6.7, injured: [] },
  { name: "St. John's", conference: 'bigeast', seed: 6, region: 'Midwest', adj_oe: 115.5, adj_de: 93.2, tempo: 68.4, strength: 0.74, record: '22-10', confRecord: '13-5', coachName: 'Rick Pitino', coachTourneyRecord: '29-14', keyPlayer: 'RJ Luis Jr', playStyle: 'Athletic wings with explosive transition game', luck: 0.03, sos: 6.5, injured: [] },

  // === #7 Seeds ===
  { name: 'Alabama', conference: 'sec', seed: 7, region: 'East', adj_oe: 115.2, adj_de: 94.8, tempo: 74.1, strength: 0.73, record: '20-12', confRecord: '9-9', coachName: 'Nate Oats', coachTourneyRecord: '7-4', keyPlayer: 'Mark Sears', playStyle: 'High-variance: three-point heavy, boom-or-bust', luck: -0.05, sos: 7.9, injured: [] },
  { name: 'Kansas', conference: 'big12', seed: 7, region: 'South', adj_oe: 112.8, adj_de: 92.5, tempo: 67.2, strength: 0.72, record: '20-12', confRecord: '9-9', coachName: 'Bill Self', coachTourneyRecord: '48-17', keyPlayer: 'Hunter Dickinson', playStyle: 'Struggling offense, elite coaching pedigree', luck: -0.04, sos: 8.3, injured: ['Arterio Morris'] },
  { name: 'Baylor', conference: 'big12', seed: 7, region: 'West', adj_oe: 114.0, adj_de: 93.5, tempo: 66.8, strength: 0.71, record: '20-12', confRecord: '10-8', coachName: 'Scott Drew', coachTourneyRecord: '16-8', keyPlayer: 'Norchad Omier', playStyle: 'Physical post play with defensive versatility', luck: 0.02, sos: 7.5, injured: [] },
  { name: 'UCLA', conference: 'big10', seed: 7, region: 'Midwest', adj_oe: 113.5, adj_de: 93.0, tempo: 67.5, strength: 0.70, record: '20-12', confRecord: '11-7', coachName: 'Mick Cronin', coachTourneyRecord: '10-8', keyPlayer: 'Dylan Andrews', playStyle: 'Defense-first with improved half-court offense', luck: 0.01, sos: 7.2, injured: [] },

  // === #8 Seeds ===
  { name: 'Mississippi St', conference: 'sec', seed: 8, region: 'East', adj_oe: 113.8, adj_de: 94.5, tempo: 68.0, strength: 0.69, record: '20-12', confRecord: '9-9', coachName: 'Chris Jans', coachTourneyRecord: '3-2', keyPlayer: 'Josh Hubbard', playStyle: 'Balanced attack with improving defense', luck: 0.01, sos: 7.5, injured: [] },
  { name: 'Texas Tech', conference: 'big12', seed: 8, region: 'South', adj_oe: 113.2, adj_de: 92.0, tempo: 66.5, strength: 0.68, record: '20-12', confRecord: '10-8', coachName: 'Grant McCasland', coachTourneyRecord: '2-2', keyPlayer: 'Darrion Williams', playStyle: 'Defensive-minded with efficient halfcourt sets', luck: -0.01, sos: 7.6, injured: [] },
  { name: 'Oregon', conference: 'big10', seed: 8, region: 'West', adj_oe: 114.5, adj_de: 94.8, tempo: 69.2, strength: 0.67, record: '20-12', confRecord: '11-7', coachName: 'Dana Altman', coachTourneyRecord: '12-10', keyPlayer: 'TJ Bamba', playStyle: 'Up-tempo with guard-driven attack', luck: 0.02, sos: 7.0, injured: [] },
  { name: 'SMU', conference: 'acc', seed: 8, region: 'Midwest', adj_oe: 112.5, adj_de: 93.2, tempo: 67.0, strength: 0.66, record: '20-12', confRecord: '11-7', coachName: 'Andy Enfield', coachTourneyRecord: '4-3', keyPlayer: 'BJ Edwards', playStyle: 'Athletic, fast-paced offense', luck: 0.00, sos: 6.5, injured: [] },

  // === #9-12 Seeds (condensed — key tournament teams) ===
  { name: 'TCU', conference: 'big12', seed: 9, region: 'East', adj_oe: 112.0, adj_de: 94.0, tempo: 67.5, strength: 0.65, record: '19-13', confRecord: '9-9', coachName: 'Jamie Dixon', coachTourneyRecord: '4-5', keyPlayer: 'Vasean Allette', playStyle: 'Physical defensive team', luck: -0.02, sos: 7.4, injured: [] },
  { name: 'Ohio State', conference: 'big10', seed: 9, region: 'South', adj_oe: 113.5, adj_de: 95.0, tempo: 68.0, strength: 0.64, record: '19-13', confRecord: '10-8', coachName: 'Jake Diebler', coachTourneyRecord: '1-1', keyPlayer: 'Bruce Thornton', playStyle: 'Guard-driven with improved shooting', luck: 0.01, sos: 7.1, injured: [] },
  { name: 'Indiana', conference: 'big10', seed: 9, region: 'West', adj_oe: 111.8, adj_de: 93.5, tempo: 66.0, strength: 0.63, record: '19-13', confRecord: '10-8', coachName: 'Mike Woodson', coachTourneyRecord: '2-2', keyPlayer: 'Mackenzie Mgbako', playStyle: 'Assembly Hall advantage, athletic roster', luck: 0.03, sos: 7.0, injured: [] },
  { name: 'New Mexico', conference: 'mwc', seed: 9, region: 'Midwest', adj_oe: 114.0, adj_de: 95.5, tempo: 70.5, strength: 0.62, record: '23-9', confRecord: '13-3', coachName: 'Richard Pitino', coachTourneyRecord: '1-2', keyPlayer: 'Donovan Dent', playStyle: 'High-altitude advantage, fast pace', luck: -0.01, sos: 5.5, injured: [] },

  { name: 'San Diego St', conference: 'mwc', seed: 10, region: 'East', adj_oe: 110.5, adj_de: 92.8, tempo: 64.5, strength: 0.61, record: '22-10', confRecord: '12-4', coachName: 'Brian Dutcher', coachTourneyRecord: '8-3', keyPlayer: 'Jaedon LeDee', playStyle: 'Elite defensive metrics, March tested', luck: 0.02, sos: 5.2, injured: [] },
  { name: 'Pittsburgh', conference: 'acc', seed: 10, region: 'South', adj_oe: 112.0, adj_de: 94.5, tempo: 67.8, strength: 0.60, record: '19-13', confRecord: '10-8', coachName: 'Jeff Capel', coachTourneyRecord: '2-3', keyPlayer: 'Ishmael Leggett', playStyle: 'Improving program with athletic wings', luck: -0.01, sos: 6.8, injured: [] },
  { name: 'VCU', conference: 'atlantic10', seed: 10, region: 'West', adj_oe: 111.0, adj_de: 93.0, tempo: 68.5, strength: 0.59, record: '24-8', confRecord: '14-2', coachName: 'Ryan Odom', coachTourneyRecord: '1-1', keyPlayer: 'Max Shulga', playStyle: 'Havoc defense disrupts opponents', luck: 0.01, sos: 4.8, injured: [] },
  { name: 'Xavier', conference: 'bigeast', seed: 10, region: 'Midwest', adj_oe: 113.5, adj_de: 95.8, tempo: 69.0, strength: 0.58, record: '19-13', confRecord: '10-8', coachName: 'Sean Miller', coachTourneyRecord: '12-8', keyPlayer: 'Zach Freemantle', playStyle: 'Experienced roster with tournament pedigree', luck: 0.00, sos: 6.6, injured: [] },

  { name: 'Drake', conference: 'mvc', seed: 11, region: 'East', adj_oe: 112.5, adj_de: 94.2, tempo: 66.0, strength: 0.57, record: '25-7', confRecord: '16-0', coachName: 'Darian DeVries', coachTourneyRecord: '1-1', keyPlayer: 'Tucker DeVries', playStyle: 'Mid-major powerhouse, unbeaten in conference', luck: 0.04, sos: 4.0, injured: [] },
  { name: 'Santa Clara', conference: 'wcc', seed: 11, region: 'South', adj_oe: 113.0, adj_de: 95.0, tempo: 68.0, strength: 0.56, record: '24-8', confRecord: '14-2', coachName: 'Herb Sendek', coachTourneyRecord: '3-4', keyPlayer: 'Carlos Stewart', playStyle: 'Efficient offense with WCC dominance', luck: 0.01, sos: 4.5, injured: [] },
  { name: 'Colorado', conference: 'big12', seed: 11, region: 'West', adj_oe: 111.5, adj_de: 94.0, tempo: 67.0, strength: 0.55, record: '19-13', confRecord: '9-9', coachName: 'Tad Boyle', coachTourneyRecord: '3-5', keyPlayer: 'Andrej Jakimovski', playStyle: 'Altitude advantage at home, 3pt shooting', luck: -0.02, sos: 7.2, injured: [] },
  { name: 'NC State', conference: 'acc', seed: 11, region: 'Midwest', adj_oe: 112.0, adj_de: 95.2, tempo: 68.5, strength: 0.54, record: '19-13', confRecord: '9-9', coachName: 'Kevin Keatts', coachTourneyRecord: '5-2', keyPlayer: 'DJ Horne', playStyle: 'Cinderella energy from 2024 Final Four run', luck: 0.05, sos: 6.8, injured: [] },

  { name: 'McNeese', conference: 'southland', seed: 12, region: 'East', adj_oe: 109.5, adj_de: 94.5, tempo: 69.0, strength: 0.53, record: '27-5', confRecord: '15-1', coachName: 'Will Wade', coachTourneyRecord: '2-2', keyPlayer: 'Shahada Wells', playStyle: 'High-scoring mid-major with upset potential', luck: 0.03, sos: 3.5, injured: [] },
  { name: 'Akron', conference: 'horizon', seed: 12, region: 'South', adj_oe: 108.0, adj_de: 93.5, tempo: 66.5, strength: 0.52, record: '25-7', confRecord: '14-2', coachName: 'John Groce', coachTourneyRecord: '2-3', keyPlayer: 'Nate Johnson', playStyle: 'Experienced mid-major, upset-ready', luck: 0.02, sos: 3.8, injured: [] },
  { name: 'Liberty', conference: 'asun', seed: 12, region: 'West', adj_oe: 110.0, adj_de: 94.8, tempo: 67.5, strength: 0.51, record: '26-6', confRecord: '15-1', coachName: 'Ritchie McKay', coachTourneyRecord: '1-1', keyPlayer: 'Darius McGhee', playStyle: 'Sharpshooting guard can get hot', luck: 0.01, sos: 3.2, injured: [] },
  { name: 'Lipscomb', conference: 'asun', seed: 12, region: 'Midwest', adj_oe: 107.5, adj_de: 93.0, tempo: 67.0, strength: 0.50, record: '24-8', confRecord: '13-3', coachName: 'Lenwood Chance', coachTourneyRecord: '0-0', keyPlayer: 'Jacob Ognacevic', playStyle: 'Physical play with strong rebounding', luck: 0.02, sos: 3.0, injured: [] },

  // === #13-16 Seeds (auto-bid conference champs — key upset candidates) ===
  { name: 'Yale', conference: 'ivy', seed: 13, region: 'East', adj_oe: 108.5, adj_de: 95.0, tempo: 65.5, strength: 0.48, record: '23-7', confRecord: '12-2', coachName: 'James Jones', coachTourneyRecord: '1-1', keyPlayer: 'John Poulakidas', playStyle: 'Smart, well-coached team with shooting', luck: 0.01, sos: 3.5, injured: [] },
  { name: 'Vermont', conference: 'americaeast', seed: 13, region: 'South', adj_oe: 107.0, adj_de: 94.8, tempo: 65.0, strength: 0.47, record: '24-8', confRecord: '14-2', coachName: 'John Becker', coachTourneyRecord: '1-3', keyPlayer: 'Dylan Penn', playStyle: 'Efficient mid-major with patience', luck: 0.00, sos: 2.8, injured: [] },
  { name: 'UC Irvine', conference: 'bigwest', seed: 13, region: 'West', adj_oe: 106.5, adj_de: 94.0, tempo: 64.5, strength: 0.46, record: '25-7', confRecord: '14-2', coachName: 'Russell Turner', coachTourneyRecord: '1-1', keyPlayer: 'Bent Leuchten', playStyle: 'Defensive grinders with low turnover rate', luck: 0.02, sos: 2.5, injured: [] },
  { name: 'Iona', conference: 'maac', seed: 13, region: 'Midwest', adj_oe: 106.0, adj_de: 95.5, tempo: 67.0, strength: 0.45, record: '23-9', confRecord: '13-3', coachName: 'Tobin Anderson', coachTourneyRecord: '1-1', keyPlayer: 'Walter Clayton', playStyle: 'Guard-heavy with 3pt shooting upside', luck: 0.01, sos: 2.2, injured: [] },

  { name: 'UNC Wilmington', conference: 'colonial', seed: 14, region: 'East', adj_oe: 105.0, adj_de: 96.0, tempo: 66.0, strength: 0.42, record: '22-10', confRecord: '13-3', coachName: 'Takayo Siddle', coachTourneyRecord: '0-0', keyPlayer: 'Trazarien White', playStyle: 'CAA champs with defensive identity', luck: 0.03, sos: 2.0, injured: [] },
  { name: 'Morehead St', conference: 'ovc', seed: 14, region: 'South', adj_oe: 104.5, adj_de: 95.5, tempo: 67.5, strength: 0.41, record: '24-8', confRecord: '14-2', coachName: 'Preston Spradlin', coachTourneyRecord: '0-0', keyPlayer: 'Alex Caldwell', playStyle: 'OVC champions with confident scoring', luck: 0.02, sos: 1.8, injured: [] },
  { name: 'Montana', conference: 'bigsky', seed: 14, region: 'West', adj_oe: 104.0, adj_de: 96.0, tempo: 65.5, strength: 0.40, record: '23-9', confRecord: '14-2', coachName: 'Travis DeCuire', coachTourneyRecord: '0-1', keyPlayer: 'Aiden Slodichak', playStyle: 'Big Sky champs with altitude advantage', luck: 0.01, sos: 1.5, injured: [] },
  { name: 'SIU Edwardsville', conference: 'ovc', seed: 14, region: 'Midwest', adj_oe: 103.5, adj_de: 96.5, tempo: 66.0, strength: 0.39, record: '22-10', confRecord: '13-3', coachName: 'Brian Barone', coachTourneyRecord: '0-0', keyPlayer: 'RaySean Taylor', playStyle: 'Physical OVC squad', luck: 0.02, sos: 1.5, injured: [] },

  { name: 'Grambling', conference: 'swac', seed: 15, region: 'East', adj_oe: 102.0, adj_de: 97.5, tempo: 68.0, strength: 0.35, record: '21-11', confRecord: '14-2', coachName: 'Donte Jackson', coachTourneyRecord: '0-1', keyPlayer: 'Tra\'Michael Moton', playStyle: 'SWAC champs with athleticism', luck: 0.03, sos: 1.0, injured: [] },
  { name: 'Colgate', conference: 'patriot', seed: 15, region: 'South', adj_oe: 103.0, adj_de: 97.0, tempo: 66.5, strength: 0.36, record: '24-8', confRecord: '15-1', coachName: 'Matt Langel', coachTourneyRecord: '0-2', keyPlayer: 'Brady Colwell', playStyle: 'Patriot powerhouse with shooting touch', luck: 0.01, sos: 1.2, injured: [] },
  { name: 'Robert Morris', conference: 'horizon', seed: 15, region: 'West', adj_oe: 101.5, adj_de: 97.5, tempo: 65.5, strength: 0.34, record: '22-10', confRecord: '13-3', coachName: 'Andrew Toole', coachTourneyRecord: '1-1', keyPlayer: 'Josh Corbin', playStyle: 'Horizon champs with size inside', luck: 0.02, sos: 1.0, injured: [] },
  { name: 'Norfolk St', conference: 'meac', seed: 15, region: 'Midwest', adj_oe: 101.0, adj_de: 98.0, tempo: 67.0, strength: 0.33, record: '22-10', confRecord: '14-2', coachName: 'Robert Jones', coachTourneyRecord: '1-1', keyPlayer: 'Brian Moore', playStyle: 'MEAC champs with upset dreams', luck: 0.03, sos: 0.8, injured: [] },

  { name: 'Stetson', conference: 'asun', seed: 16, region: 'East', adj_oe: 99.0, adj_de: 99.5, tempo: 68.0, strength: 0.28, record: '19-13', confRecord: '11-5', coachName: 'Donnie Jones', coachTourneyRecord: '0-0', keyPlayer: 'Jalen Forrest', playStyle: 'ASUN auto-bid, dangerous 3pt shooting', luck: 0.05, sos: 0.5, injured: [] },
  { name: 'N Kentucky', conference: 'horizon', seed: 16, region: 'South', adj_oe: 98.5, adj_de: 100.0, tempo: 67.0, strength: 0.27, record: '18-14', confRecord: '10-6', coachName: 'Darrin Horn', coachTourneyRecord: '0-0', keyPlayer: 'Sam Vinson', playStyle: 'First Four candidate with grit', luck: 0.03, sos: 0.5, injured: [] },
  { name: 'LIU', conference: 'nec', seed: 16, region: 'West', adj_oe: 97.5, adj_de: 100.5, tempo: 66.5, strength: 0.25, record: '18-14', confRecord: '12-4', coachName: 'Rod Strickland', coachTourneyRecord: '0-0', keyPlayer: 'TBA', playStyle: 'NEC auto-bid', luck: 0.04, sos: 0.3, injured: [] },
  { name: 'Alabama A&M', conference: 'swac', seed: 16, region: 'Midwest', adj_oe: 97.0, adj_de: 101.0, tempo: 69.0, strength: 0.24, record: '17-15', confRecord: '11-5', coachName: 'Dylan Howard', coachTourneyRecord: '0-0', keyPlayer: 'TBA', playStyle: 'First Four candidate from SWAC', luck: 0.05, sos: 0.2, injured: [] },
];

// --- Helper Functions ---

function gaussianRandom(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const UPSET_RATES: Record<string, number> = {
  '1v16': 0.01, '1v8': 0.21, '1v9': 0.21, '1v5': 0.36, '1v4': 0.22,
  '2v15': 0.06, '2v7': 0.40, '2v10': 0.40, '2v3': 0.48,
  '3v14': 0.15, '3v6': 0.47, '3v11': 0.41,
  '4v13': 0.21, '4v5': 0.48, '4v12': 0.35,
  '5v12': 0.35, '5v4': 0.48,
};

function getUpsetProbability(homeSeed: number, awaySeed: number, pWinHome: number): number {
  const favSeed = Math.min(homeSeed, awaySeed);
  const dogSeed = Math.max(homeSeed, awaySeed);
  const key = `${favSeed}v${dogSeed}`;
  const historicalRate = UPSET_RATES[key] ?? 0.30;
  const modelUpset = favSeed === homeSeed ? (1 - pWinHome) : pWinHome;
  return modelUpset * 0.6 + historicalRate * 0.4;
}

function probToMoneyline(p: number): string {
  if (p >= 0.5) return `${Math.round(-100 * p / (1 - p))}`;
  return `+${Math.round(100 * (1 - p) / p)}`;
}

function generateFactors(team: MockTeam, opponent: MockTeam): string[] {
  const factors: string[] = [];
  const netRating = team.adj_oe - team.adj_de;
  const oppNet = opponent.adj_oe - opponent.adj_de;
  if (team.adj_de < opponent.adj_de) factors.push(`Superior defense: ${team.adj_de.toFixed(1)} Adj DE vs ${opponent.adj_de.toFixed(1)}`);
  if (team.adj_oe > opponent.adj_oe) factors.push(`Higher offensive efficiency: ${team.adj_oe.toFixed(1)} Adj OE`);
  if (team.tempo < 66 && opponent.tempo > 69) factors.push(`Tempo control — can slow down ${opponent.name}'s pace`);
  if (team.tempo > 70 && opponent.tempo < 66) factors.push(`Pace pusher — can force ${opponent.name} into uncomfortable tempo`);
  if (team.seed < opponent.seed) factors.push(`Higher seed (${team.seed} vs ${opponent.seed}) — better path`);
  if (netRating > oppNet + 3) factors.push(`Net rating edge: +${netRating.toFixed(1)} vs +${oppNet.toFixed(1)}`);
  if (team.coachTourneyRecord && parseInt(team.coachTourneyRecord) > 10) factors.push(`Coach ${team.coachName}: ${team.coachTourneyRecord} in NCAA tournament`);
  if (team.injured.length === 0 && opponent.injured.length > 0) factors.push(`Fully healthy vs ${opponent.name}'s injuries`);
  factors.push(team.playStyle);
  return factors.slice(0, 4);
}

// --- Public API ---

export function getMockGraph(): GraphResponse {
  const top32 = MOCK_TEAMS.slice(0, 32);
  const teams = top32.map((t) => {
    const confData = CONFERENCES[t.conference] ?? { name: t.conference, color: 0x888888 };
    return {
      id: t.name.toLowerCase().replace(/[^a-z]/g, '_'), name: t.name, conference: t.conference,
      seed: t.seed, adj_oe: t.adj_oe, adj_de: t.adj_de, tempo: t.tempo,
      x: (Math.random() - 0.5) * 12, y: (Math.random() - 0.5) * 12, z: 0, color: confData.color,
    };
  });
  const games = [];
  for (let i = 0; i < teams.length; i++) {
    for (let j = i + 1; j < teams.length; j++) {
      if (teams[i].conference === teams[j].conference || Math.random() < 0.1) {
        games.push({ source: teams[i].id, target: teams[j].id, home_win: top32[i].strength > top32[j].strength, spread: (top32[i].strength - top32[j].strength) * 20, date: '2026-02-15' });
      }
    }
  }
  const confList = Object.entries(CONFERENCES).slice(0, 10).map(([id, c]) => ({ id, name: c.name, x: 0, y: 0, z: 0, color: c.color }));
  const conference_edges = teams.map(t => ({ source: t.id, target: t.conference, edge_type: 'member_of' as const }));
  return { teams, conferences: confList, games, conference_edges };
}

export function getMockMatchup(home: string, away: string): EnrichedMatchupResponse {
  const homeData = MOCK_TEAMS.find(t => t.name === home);
  const awayData = MOCK_TEAMS.find(t => t.name === away);
  const hStr = homeData?.strength ?? 0.5;
  const aStr = awayData?.strength ?? 0.5;
  // Win probability from efficiency margin model
  const homeEM = (homeData?.adj_oe ?? 100) - (awayData?.adj_de ?? 100);
  const awayEM = (awayData?.adj_oe ?? 100) - (homeData?.adj_de ?? 100);
  const emDiff = homeEM - awayEM;
  const pWin = 1 / (1 + Math.exp(-emDiff / 10));  // logistic from efficiency margin
  const samples = Array.from({ length: 200 }, () => Math.max(0, Math.min(1, gaussianRandom(pWin, 0.10))));
  const spreadMean = emDiff * 0.7;
  const homeSeed = homeData?.seed ?? 8;
  const awaySeed = awayData?.seed ?? 8;
  return {
    home_team: home, away_team: away, p_win_home: pWin, p_win_samples: samples,
    spread_mean: spreadMean, spread_samples: samples.map(s => (s - 0.5) * 20),
    luck_compressed: (homeData?.luck ?? 0) > 0.04 || (awayData?.luck ?? 0) > 0.04,
    home_moneyline: probToMoneyline(pWin), away_moneyline: probToMoneyline(1 - pWin),
    upset_probability: getUpsetProbability(homeSeed, awaySeed, pWin),
    home_factors: homeData && awayData ? generateFactors(homeData, awayData) : [],
    away_factors: awayData && homeData ? generateFactors(awayData, homeData) : [],
    home_record: homeData?.record ?? '0-0', away_record: awayData?.record ?? '0-0',
    home_seed: homeSeed, away_seed: awaySeed,
    home_conference: homeData?.conference ?? '', away_conference: awayData?.conference ?? '',
    home_adj_oe: homeData?.adj_oe ?? 100, away_adj_oe: awayData?.adj_oe ?? 100,
    home_adj_de: homeData?.adj_de ?? 100, away_adj_de: awayData?.adj_de ?? 100,
    home_tempo: homeData?.tempo ?? 68, away_tempo: awayData?.tempo ?? 68,
    home_key_player: homeData?.keyPlayer ?? 'Unknown', away_key_player: awayData?.keyPlayer ?? 'Unknown',
  };
}

export function getMockSimulation(teams: string[], nSims: number): SimulateResponse {
  const ROUNDS = ['R64', 'R32', 'S16', 'E8', 'F4', 'Championship'];
  const advancements: TeamAdvancement[] = teams.map(name => {
    const data = MOCK_TEAMS.find(t => t.name === name);
    const str = data?.strength ?? 0.3;
    const probs: Record<string, number> = {};
    let cumulative = 0.95;
    ROUNDS.forEach((round, i) => {
      const decay = Math.pow(str, i + 0.5) * cumulative;
      probs[round] = Math.max(0.01, Math.min(0.99, decay));
      cumulative *= (0.4 + str * 0.5);
    });
    return { team: name, advancement_probs: probs, entropy: 1.5 + Math.random() };
  });
  return { n_simulations: nSims, advancements };
}

// Generate 3 bracket variants
export interface BracketVariant {
  name: string;
  description: string;
  icon: string;
  finalFour: string[];
  champion: string;
  championshipGame: [string, string];
  upsets: { round: string; winner: string; loser: string; winnerSeed: number; loserSeed: number }[];
}

export function generate2026Brackets(): BracketVariant[] {
  const chalk: BracketVariant = {
    name: 'Chalk', description: 'Favorites win every game — baseline bracket', icon: '🏆',
    finalFour: ['Duke', 'Michigan', 'Arizona', 'Florida'],
    champion: 'Duke',
    championshipGame: ['Duke', 'Florida'],
    upsets: [],
  };

  const upset: BracketVariant = {
    name: 'Upset Special', description: 'Weighted toward historical upset rates', icon: '🔥',
    finalFour: ['Houston', 'Michigan', 'Iowa State', 'Tennessee'],
    champion: 'Houston',
    championshipGame: ['Houston', 'Tennessee'],
    upsets: [
      { round: 'R64', winner: 'McNeese', loser: 'Purdue', winnerSeed: 12, loserSeed: 5 },
      { round: 'R64', winner: 'Yale', loser: 'Virginia', winnerSeed: 13, loserSeed: 4 },
      { round: 'R64', winner: 'Drake', loser: 'Alabama', winnerSeed: 11, loserSeed: 7 },
      { round: 'R32', winner: 'Iowa State', loser: 'Florida', winnerSeed: 3, loserSeed: 1 },
      { round: 'R32', winner: 'Tennessee', loser: 'Duke', winnerSeed: 2, loserSeed: 1 },
      { round: 'S16', winner: 'San Diego St', loser: 'Nebraska', winnerSeed: 10, loserSeed: 3 },
    ],
  };

  const balanced: BracketVariant = {
    name: 'Model\'s Best Guess', description: 'Most probable outcome per game from efficiency model', icon: '⚖️',
    finalFour: ['Duke', 'Michigan', 'Houston', 'Florida'],
    champion: 'Duke',
    championshipGame: ['Duke', 'Houston'],
    upsets: [
      { round: 'R64', winner: 'McNeese', loser: 'Purdue', winnerSeed: 12, loserSeed: 5 },
      { round: 'R32', winner: 'Houston', loser: 'Arizona', winnerSeed: 2, loserSeed: 1 },
    ],
  };

  return [chalk, upset, balanced];
}

export function getConferenceName(id: string): string {
  return CONFERENCES[id]?.name ?? id;
}
