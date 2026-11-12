import { DemoClient } from './demo.js';

export class ClientError extends Error {
  constructor(message, code = 'request_failed', status = 400) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

class ApiClient {
  constructor(baseUrl) { this.baseUrl = baseUrl.replace(/\/$/, ''); }
  async request(path, body, key) {
    const token = sessionStorage.getItem('rush-labeling-token');
    const headers = { 'Accept': 'application/json' };
    if (token) headers.Authorization = `Bearer ${token}`;
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      headers['Idempotency-Key'] = key || crypto.randomUUID();
    }
    let response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, { method: body === undefined ? 'GET' : 'POST', headers, body: body === undefined ? undefined : JSON.stringify(body), credentials: 'same-origin' });
    } catch {
      throw new ClientError('The review service is unreachable. Your unsaved choices are still here.', 'network_error', 0);
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new ClientError(payload.error?.message || `Request failed (${response.status}).`, payload.error?.code, response.status);
    return payload;
  }
  bootstrap() { return this.request('/bootstrap'); }
  items(filters = {}) { const query = new URLSearchParams(Object.entries(filters).filter(([,v]) => v)); return this.request(`/items?${query}`); }
  item(id) { return this.request(`/items/${encodeURIComponent(id)}`); }
  review(id, payload, key) { return this.request(`/items/${encodeURIComponent(id)}/reviews`, payload, key); }
  claim(id, payload, key) { return this.request(`/items/${encodeURIComponent(id)}/claim`, payload, key); }
  route(id, payload, key) { return this.request(`/items/${encodeURIComponent(id)}/route`, payload, key); }
  publish(payload, key) { return this.request('/epochs', payload, key); }
  rerun(payload, key) { return this.request('/reruns', payload, key); }
  exportEpoch(epoch) { return this.request(`/epochs/${encodeURIComponent(epoch)}/export`); }
  events() { return this.request('/events'); }
  importItems(items, key) { return this.request('/items/import', {items}, key); }
}

export async function createClient() {
  const config = window.RUSH_LABELING || {mode: 'demo'};
  if (config.mode === 'api') return new ApiClient(config.baseUrl || '/api/labeling');
  const client = new DemoClient();
  await client.init();
  return client;
}
