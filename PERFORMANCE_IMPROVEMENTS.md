# Performance Improvements

Last updated: 2026-02-15

This file separates:
- `Current`: what exists in code today
- `Proposed`: improvements not yet implemented

## Current

- FastAPI backend with async endpoints.
- Frontend built with Vite.
- Local validation script (`validate.ps1`) for setup checks.
- Health endpoint available at `/api/health`.

## Proposed Backend Improvements

1. Add caching for repeated analysis results (Redis or in-memory cache).
2. Add MongoDB indexes for high-traffic query paths.
3. Move long-running analysis tasks to background workers.
4. Add request timing and structured logging.
5. Add response compression middleware.

## Proposed Frontend Improvements

1. Route-level code splitting with lazy loading.
2. Memoization for expensive dashboard components.
3. Request-level caching and deduplication.
4. Bundle analysis and chunk tuning for build output.
5. Virtualized rendering for large history lists.

## Measurement Plan

Track these before/after metrics:
- API latency (`p50`, `p95`)
- Full document analysis duration
- Frontend first-load time
- Bundle size
- Memory usage during analysis

## Priority Order

1. Measurement baseline
2. Database indexes + caching
3. Frontend chunk optimization
4. Background task queue
5. Continuous monitoring
