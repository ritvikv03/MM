import { createClient } from '@supabase/supabase-js';

const url  = process.env.NEXT_PUBLIC_SUPABASE_URL;
const key  = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

// Returns null when env vars are absent (local dev without Supabase configured).
// All query hooks check for null and return empty data gracefully.
export const supabase = url && key ? createClient(url, key) : null;
