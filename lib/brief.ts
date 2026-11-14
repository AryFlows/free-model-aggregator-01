// The trial brief: the provider directory, the quality table, the workload manifest,
// and the catalog probe that turns the directory into the registered catalog.
//
// The brief (MAKEFUN_PROMPT_FILE) is prose carrying three JSON documents, each found by
// its `schema` member:
//   providers  free-model-aggregator-01/providers/v1  the ten providers: id, wire,
//              auth scheme, key variable, key, base path under $MAKEFUN_ENV_URL
//   ratings    free-model-aggregator-01/ratings/v1     every free model's rating in
//              milli-points, context length, tools/vision/JSON support, free flag,
//              and the best rating per request
//   workload   free-model-aggregator-01/workload/v1    the twenty requests in order:
//              id, api, stream, weight, requirements
//
// The catalog is built by probing every provider's own list route in its own wire and
// joining the listing with the quality table (the sealed truth for free flags and
// capabilities where it has a row). The catalog's base_url is the directory's path
// (/p/<id>), never joined to the env origin: the origin differs between a live pass
// and its replay, and the catalog is part of the state export a replay must reproduce.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** Any JSON object. */
export type Json = Record<string, unknown>;
export function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Base path (or absolute URL) the provider's routes hang off. */
  base: string;
  keyEnv: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  workload: WorkloadEntry[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface CatalogDoc {
  providers: CatalogProvider[];
}

/** One model as a provider's list route spelled it. */
export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

/** What a model can do, quality table first, listing second. */
export interface Capabilities {
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBoolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** Join a base path to the env origin; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const root = envUrl.replace(/\/+$/, '');
  if (b === '') return root;
  return `${root}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers carrying a provider's key, per its auth scheme. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  switch (auth) {
    case 'bearer':
      return { authorization: `Bearer ${key}` };
    case 'x-api-key':
      return { 'x-api-key': key };
    case 'x-goog-api-key':
      return { 'x-goog-api-key': key };
    default:
      return {};
  }
}

/** Index of the `}` closing the object opened at `start`, skipping string contents; -1 if none. */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i++;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function schemaDocuments(text: string): Json[] {
  const docs: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closingBrace(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') docs.push(value);
    } catch {
      // Not a standalone JSON object; keep scanning.
    }
  }
  return docs;
}

function parseProvider(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth: AuthScheme =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth,
    base,
    keyEnv: asString(v.key_env),
    key: asString(v.key),
  };
}

function parseRating(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    contextLength: asNumberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function parseWorkloadEntry(v: unknown): WorkloadEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  const requires: Requirement = {};
  if (req.tools === true) requires.tools = true;
  if (req.vision === true) requires.vision = true;
  if (req.json === true) requires.json = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) requires.minContext = req.min_context;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires,
  };
}

/** Read and parse the brief at `path`. A missing or unreadable brief yields empty tables plus an error. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (dirDoc === undefined || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = parseProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings: RatingRow[] = [];
  const best = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.ratings)) {
    for (const v of ratingsDoc.ratings) {
      const r = parseRating(v);
      if (r !== null) ratings.push(r);
    }
    if (Array.isArray(ratingsDoc.best)) {
      for (const v of ratingsDoc.best) {
        if (isRecord(v) && asString(v.id) !== '') best.set(asString(v.id), asNumberOrNull(v.rating_milli) ?? 0);
      }
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const workload: WorkloadEntry[] = [];
  if (workloadDoc !== undefined && Array.isArray(workloadDoc.manifest)) {
    for (const v of workloadDoc.manifest) {
      const w = parseWorkloadEntry(v);
      if (w !== null) workload.push(w);
    }
  }
  return { providers, ratings, best, workload, error: null };
}

/** The quality-table row for one provider's model, if the table has one. */
export function ratingRowFor(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a listed model is free: the quality table's row wins; else the listing's own word. */
export function isFreeModel(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** A model's capabilities: the quality table's row first, the listing's fields second. */
export function capabilitiesOf(model: ListedModel, row: RatingRow | null): Capabilities {
  return {
    contextLength: row?.contextLength ?? model.contextLength ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities satisfy a request's requirements. An unknown capability never satisfies. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) return false;
  return true;
}

/** Whether every price in a listing's pricing object is zero; null when the listing says nothing. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values: number[] = [];
  for (const x of Object.values(pricing)) {
    if (typeof x === 'number') values.push(x);
    else if (typeof x === 'string' && x.trim() !== '') values.push(Number(x));
  }
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing spelled it, in its wire's shape. */
function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const context =
    asNumberOrNull(v.context_length) ?? asNumberOrNull(v.context_window) ?? asNumberOrNull(v.inputTokenLimit);
  return {
    id,
    displayName:
      asString(v.display_name) !== ''
        ? asString(v.display_name)
        : asString(v.displayName) !== ''
          ? asString(v.displayName)
          : id,
    free: asBoolOrNull(v.free) ?? zeroPriced(v),
    contextLength: context,
    tools: asBoolOrNull(v.tools),
    vision: asBoolOrNull(v.vision),
    json: asBoolOrNull(v.json),
  };
}

/** The models a catalog answer listed, in its wire's shape. */
function listedModels(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModel(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's list route in its own wire. Any failure leaves the provider unlisted. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: listedModels(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byModelId = (a: ListedModel, b: ListedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The catalog: every provider that answered, in brief order, with every model it listed
 * (free and paid), sorted by id and deduplicated. A model's free flag and context length
 * come from the quality table where it has a row, else from the listing.
 */
export function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): CatalogDoc {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byModelId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowFor(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: isFreeModel(model, row),
        context_length: row?.contextLength ?? model.contextLength ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}
