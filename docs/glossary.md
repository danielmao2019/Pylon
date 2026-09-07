# Glossary <!-- omit in toc -->

This glossary governs the Pylon library layer — the code outside `project/`, the docs and skeletons that design it, and our discussion of it. Inside that scope we write and speak only the terms recorded here plus plain non-technical English; a new term enters the common ground only when one of us proposes it explicitly and the other accepts, and it is recorded here in the same turn it is accepted. An entry pins what a name names; the rules and decisions about the thing live in the doc that owns it.

## Table of Contents <!-- omit in toc -->

- [1. Spaces](#1-spaces)

----------

## 1. Spaces

- **pixel space** — one image's own 2D grid, indexed by pixel.
- **UV space** — one mesh's own 2D parameter square, indexed by UV coordinate.
- **uv validity mask** — the mask over one mesh topology's UV space separating the regions where its UV chart is valid from the ones where it is not.
