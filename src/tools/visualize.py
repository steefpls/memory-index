"""Graph visualization tool — generates interactive HTML with Cytoscape.js."""

import json
import logging
import os
import sys
import tempfile
import webbrowser
from pathlib import Path

from src.graph.manager import get_graph, get_all_relations
from src.graph.traversal import detect_communities
from src.indexer.store import get_entity, get_observations, list_entities

logger = logging.getLogger(__name__)

# Color palette by entity type
TYPE_COLORS = {
    "person": "#4FC3F7",
    "project": "#81C784",
    "concept": "#FFB74D",
    "decision": "#E57373",
    "error": "#F44336",
    "solution": "#66BB6A",
    "technology": "#BA68C8",
    "pattern": "#FFD54F",
    "preference": "#FF8A65",
    "organization": "#4DD0E1",
    "event": "#AED581",
    "reference": "#90A4AE",
}
DEFAULT_COLOR = "#B0BEC5"

# Community hull colors (semi-transparent)
COMMUNITY_COLORS = [
    "rgba(79, 195, 247, 0.08)",
    "rgba(129, 199, 132, 0.08)",
    "rgba(255, 183, 77, 0.08)",
    "rgba(186, 104, 200, 0.08)",
    "rgba(77, 208, 225, 0.08)",
    "rgba(233, 69, 96, 0.08)",
    "rgba(174, 213, 129, 0.08)",
    "rgba(255, 138, 101, 0.08)",
]
COMMUNITY_BORDER_COLORS = [
    "rgba(79, 195, 247, 0.3)",
    "rgba(129, 199, 132, 0.3)",
    "rgba(255, 183, 77, 0.3)",
    "rgba(186, 104, 200, 0.3)",
    "rgba(77, 208, 225, 0.3)",
    "rgba(233, 69, 96, 0.3)",
    "rgba(174, 213, 129, 0.3)",
    "rgba(255, 138, 101, 0.3)",
]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>memory-index — Knowledge Graph</title>
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v3.min.js"></script>
<script src="https://d3js.org/d3-quadtree.v3.min.js"></script>
<script src="https://d3js.org/d3-timer.v3.min.js"></script>
<script src="https://d3js.org/d3-force.v3.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f0f1a; color: #e0e0e0; overflow: hidden;
  }
  #cy { width: 100vw; height: 100vh; }

  /* Overlays share this base */
  .overlay {
    position: fixed; background: rgba(16, 18, 36, 0.92);
    backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; z-index: 10; font-size: 0.82em;
  }

  /* Top bar */
  #topbar {
    top: 16px; left: 16px; right: 16px; height: auto;
    padding: 10px 16px; display: flex; align-items: center; gap: 16px;
    flex-wrap: wrap;
  }
  #topbar .title { font-weight: 700; color: #e94560; font-size: 1.05em; white-space: nowrap; }
  #topbar .stat { color: #888; white-space: nowrap; }
  #topbar .sep { width: 1px; height: 18px; background: rgba(255,255,255,0.1); }
  #topbar .filters { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  #topbar .filter-btn {
    padding: 3px 10px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.12);
    background: transparent; color: #ccc; cursor: pointer; font-size: 0.85em;
    transition: all 0.15s;
  }
  #topbar .filter-btn:hover { border-color: rgba(255,255,255,0.3); }
  #topbar .filter-btn.active { background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.25); color: #fff; }
  #topbar .filter-btn .dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 5px; vertical-align: middle;
  }
  #topbar label { color: #666; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; }

  /* Side panel */
  #panel {
    position: fixed; top: 0; right: -440px; width: 420px; height: 100vh;
    background: rgba(16, 18, 36, 0.96); backdrop-filter: blur(16px);
    border-left: 1px solid rgba(255,255,255,0.06);
    padding: 28px 24px; overflow-y: auto; transition: right 0.3s cubic-bezier(.4,0,.2,1);
    z-index: 20;
  }
  #panel.open { right: 0; }
  #panel h2 { color: #fff; margin-bottom: 6px; font-size: 1.25em; font-weight: 600; }
  #panel .type-badge {
    display: inline-block; padding: 2px 12px; border-radius: 12px;
    font-size: 0.75em; color: #0f0f1a; font-weight: 600;
  }
  #panel .entity-id { margin-top: 6px; font-size: 0.75em; color: #555; font-family: monospace; }
  #panel .section { margin-top: 20px; }
  #panel .section h3 {
    font-size: 0.75em; text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 10px; color: #4FC3F7; font-weight: 600;
  }
  #panel .obs {
    background: rgba(255,255,255,0.04); padding: 10px 14px; border-radius: 8px;
    margin-bottom: 6px; font-size: 0.85em; line-height: 1.5;
    border-left: 2px solid rgba(79, 195, 247, 0.3);
  }
  #panel .obs .source { color: #666; font-size: 0.78em; margin-top: 4px; }
  #panel .rel {
    padding: 6px 0; font-size: 0.85em;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    cursor: pointer; transition: color 0.15s;
  }
  #panel .rel:hover { color: #4FC3F7; }
  #panel .rel:last-child { border-bottom: none; }
  #panel .close-btn {
    position: absolute; top: 16px; right: 16px; background: none; border: none;
    color: #555; font-size: 1.4em; cursor: pointer; transition: color 0.15s;
  }
  #panel .close-btn:hover { color: #e94560; }

  /* Trail breadcrumb */
  #trail {
    position: fixed; bottom: 16px; left: 16px; right: 16px;
    padding: 10px 16px; display: none; gap: 0; align-items: center;
    overflow-x: auto; white-space: nowrap;
  }
  #trail.visible { display: flex; }
  #trail .crumb {
    color: #4FC3F7; cursor: pointer; font-weight: 500;
    transition: color 0.15s; flex-shrink: 0;
  }
  #trail .crumb:hover { color: #e94560; }
  #trail .arrow { color: #444; margin: 0 8px; flex-shrink: 0; }
  #trail .relation-label { color: #666; font-size: 0.85em; margin: 0 4px; flex-shrink: 0; }
  #trail .clear-btn {
    margin-left: auto; padding: 2px 10px; border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1); background: transparent;
    color: #888; cursor: pointer; font-size: 0.8em; flex-shrink: 0;
  }
  #trail .clear-btn:hover { color: #e94560; border-color: #e94560; }

  /* Focus mode hint */
  #focus-hint {
    position: fixed; bottom: 60px; left: 50%; transform: translateX(-50%);
    padding: 6px 16px; border-radius: 20px; font-size: 0.78em;
    color: #888; display: none;
  }
  #focus-hint.visible { display: block; }
  #focus-hint kbd {
    background: rgba(255,255,255,0.08); padding: 1px 6px; border-radius: 4px;
    font-family: monospace; font-size: 0.95em; border: 1px solid rgba(255,255,255,0.1);
  }

  /* Community toggle */
  #community-toggle {
    display: flex; align-items: center; gap: 6px; cursor: pointer; color: #888;
  }
  #community-toggle input { cursor: pointer; }

  /* Canvas for hulls */
  #hull-canvas {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    pointer-events: none; z-index: 1;
  }
  #cy { z-index: 2; position: relative; }
</style>
</head>
<body>

<canvas id="hull-canvas"></canvas>
<div id="cy"></div>

<div id="topbar" class="overlay">
  <span class="title">memory-index</span>
  <span class="stat"><span id="node-count"></span> entities · <span id="edge-count"></span> relations</span>
  <span class="sep"></span>
  <label>types</label>
  <div class="filters" id="type-filters"></div>
  <span class="sep"></span>
  <label>relations</label>
  <div class="filters" id="rel-filters"></div>
  <span class="sep"></span>
  <label id="community-toggle"><input type="checkbox" id="hull-toggle"> communities</label>
</div>

<div id="panel">
  <button class="close-btn" onclick="closePanel()">&times;</button>
  <div id="panel-content"></div>
</div>

<div id="trail" class="overlay"></div>
<div id="focus-hint" class="overlay"><kbd>Esc</kbd> exit focus · <kbd>←</kbd><kbd>→</kbd> navigate neighbors</div>

<script>
const graphData = __GRAPH_DATA__;

// --- State ---
let focusMode = false;
let focusNode = null;
let focusNeighborIdx = 0;
let focusNeighbors = [];
const trail = [];  // [{id, label, relationLabel}]

// --- Stats ---
document.getElementById('node-count').textContent = graphData.nodes.length;
document.getElementById('edge-count').textContent = graphData.edges.length;

// --- Type filter buttons ---
const typesPresent = [...new Set(graphData.nodes.map(n => n.data.entity_type))].sort();
const typeFiltersEl = document.getElementById('type-filters');
const activeTypes = new Set(typesPresent);
typesPresent.forEach(t => {
  const btn = document.createElement('button');
  btn.className = 'filter-btn active';
  btn.dataset.type = t;
  const color = graphData.typeColors[t] || graphData.defaultColor;
  btn.innerHTML = '<span class="dot" style="background:' + color + '"></span>' + t;
  btn.onclick = () => toggleTypeFilter(t, btn);
  typeFiltersEl.appendChild(btn);
});

// --- Relation filter buttons ---
const relTypesPresent = [...new Set(graphData.edges.map(e => e.data.label))].sort();
const relFiltersEl = document.getElementById('rel-filters');
const activeRels = new Set(relTypesPresent);
relTypesPresent.forEach(t => {
  const btn = document.createElement('button');
  btn.className = 'filter-btn active';
  btn.dataset.rel = t;
  btn.textContent = t;
  btn.onclick = () => toggleRelFilter(t, btn);
  relFiltersEl.appendChild(btn);
});

// --- Cytoscape init ---
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: { nodes: graphData.nodes, edges: graphData.edges },
  style: [
    {
      selector: 'node',
      style: {
        'label': 'data(label)',
        'background-color': 'data(color)',
        'width': 'data(size)',
        'height': 'data(size)',
        'border-width': 0,
        'opacity': 0.85,
        'color': '#aaa',
        'text-valign': 'bottom',
        'text-margin-y': 8,
        'font-size': 'data(fontSize)',
        'text-outline-width': 2,
        'text-outline-color': '#0f0f1a',
        'transition-property': 'opacity, border-width, border-color, width, height',
        'transition-duration': '0.2s',
      }
    },
    {
      selector: 'node.hover',
      style: {
        'label': 'data(label)',
        'color': '#fff',
        'text-valign': 'bottom',
        'text-margin-y': 8,
        'font-size': '12px',
        'text-outline-width': 2,
        'text-outline-color': '#0f0f1a',
        'border-width': 2,
        'border-color': '#fff',
        'opacity': 1,
      }
    },
    {
      selector: 'edge',
      style: {
        'width': 1,
        'line-color': 'rgba(255,255,255,0.08)',
        'target-arrow-color': 'rgba(255,255,255,0.12)',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'arrow-scale': 0.7,
        'opacity': 0.6,
        'transition-property': 'opacity, line-color, width',
        'transition-duration': '0.2s',
      }
    },
    {
      selector: 'edge.hover',
      style: {
        'label': 'data(label)',
        'font-size': '9px',
        'color': '#aaa',
        'text-rotation': 'autorotate',
        'text-outline-width': 1.5,
        'text-outline-color': '#0f0f1a',
        'line-color': 'rgba(255,255,255,0.25)',
        'target-arrow-color': 'rgba(255,255,255,0.35)',
        'width': 1.5,
        'opacity': 1,
      }
    },
    {
      selector: '.highlighted',
      style: { 'opacity': 1 }
    },
    {
      selector: 'edge.highlighted',
      style: {
        'line-color': '#e94560',
        'target-arrow-color': '#e94560',
        'width': 2,
        'opacity': 1,
        'label': 'data(label)',
        'font-size': '9px',
        'color': '#e0e0e0',
        'text-rotation': 'autorotate',
        'text-outline-width': 1.5,
        'text-outline-color': '#0f0f1a',
      }
    },
    {
      selector: 'node.highlighted',
      style: {
        'label': 'data(label)',
        'color': '#fff',
        'text-valign': 'bottom',
        'text-margin-y': 8,
        'font-size': '11px',
        'text-outline-width': 2,
        'text-outline-color': '#0f0f1a',
        'opacity': 1,
        'border-width': 2,
        'border-color': '#e94560',
      }
    },
    {
      selector: '.faded',
      style: { 'opacity': 0.06 }
    },
    {
      selector: 'node.trail-node',
      style: {
        'border-width': 2,
        'border-color': '#FFB74D',
        'opacity': 1,
      }
    },
    {
      selector: 'edge.trail-edge',
      style: {
        'line-color': 'rgba(255, 183, 77, 0.5)',
        'target-arrow-color': 'rgba(255, 183, 77, 0.5)',
        'width': 2,
        'opacity': 1,
      }
    },
    {
      selector: '.hidden',
      style: { 'display': 'none' }
    },
  ],
  layout: { name: 'preset' },
  wheelSensitivity: 0.25,
  minZoom: 0.3,
  maxZoom: 3,
});

// --- D3-force physics simulation ---
// Build d3 node/link arrays from cytoscape elements
const d3Nodes = cy.nodes().map(n => ({
  id: n.id(),
  x: Math.random() * window.innerWidth,
  y: Math.random() * window.innerHeight,
  degree: n.degree(),
  size: n.data('size') || 20,
}));
const d3NodeMap = {};
d3Nodes.forEach(n => { d3NodeMap[n.id] = n; });

const d3Links = cy.edges().map(e => ({
  source: e.source().id(),
  target: e.target().id(),
}));

const simulation = d3.forceSimulation(d3Nodes)
  .force('charge', d3.forceManyBody()
    .strength(d => -250 - (d.degree * 80))
    .distanceMax(600)
  )
  .force('link', d3.forceLink(d3Links)
    .id(d => d.id)
    .distance(130)
    .strength(0.35)
  )
  .force('center', d3.forceCenter(
    window.innerWidth / 2,
    window.innerHeight / 2
  ))
  .force('collide', d3.forceCollide()
    .radius(d => d.size / 2 + 25)
    .strength(0.8)
  )
  .force('x', d3.forceX(window.innerWidth / 2).strength(0.012))
  .force('y', d3.forceY(window.innerHeight / 2).strength(0.012))
  .velocityDecay(0.35)
  .alphaDecay(0.003)
  .alphaMin(0.0005)
  .on('tick', () => {
    cy.batch(() => {
      d3Nodes.forEach(d => {
        if (d.fx != null) return; // skip dragged nodes during drag
        const node = cy.getElementById(d.id);
        if (node.length) node.position({ x: d.x, y: d.y });
      });
    });
  });

// After initial settling, fit the view
setTimeout(() => { cy.fit(undefined, 80); }, 2500);

// --- Drag: pin node while dragging, release after ---
let draggedNode = null;
cy.on('grab', 'node', function(evt) {
  const id = evt.target.id();
  const d = d3NodeMap[id];
  if (d) {
    draggedNode = d;
    d.fx = d.x;
    d.fy = d.y;
    simulation.alphaTarget(0.1).restart();
  }
});
cy.on('drag', 'node', function(evt) {
  if (draggedNode) {
    const pos = evt.target.position();
    draggedNode.fx = pos.x;
    draggedNode.fy = pos.y;
  }
});
cy.on('free', 'node', function(evt) {
  if (draggedNode) {
    // Release: let physics take over with a gentle nudge
    draggedNode.fx = null;
    draggedNode.fy = null;
    draggedNode = null;
    simulation.alphaTarget(0.05).restart();
    setTimeout(() => simulation.alphaTarget(0), 1500);
  }
});

// All labels shown by default with dynamic font sizes

// --- Hover: show label + connected edges ---
cy.on('mouseover', 'node', function(evt) {
  const node = evt.target;
  node.addClass('hover');
  node.connectedEdges().addClass('hover');
});
cy.on('mouseout', 'node', function(evt) {
  const node = evt.target;
  node.removeClass('hover');
  node.connectedEdges().removeClass('hover');
});

// --- Direct edge hover: show relation label on the edge itself ---
cy.on('mouseover', 'edge', function(evt) {
  evt.target.addClass('hover');
});
cy.on('mouseout', 'edge', function(evt) {
  evt.target.removeClass('hover');
});

// --- Single click: select + show panel + add to trail ---
cy.on('tap', 'node', function(evt) {
  const node = evt.target;
  selectNode(node);
});

// --- Double click: focus mode ---
let tapTimeout = null;
let lastTapNode = null;
cy.on('tap', 'node', function(evt) {
  const node = evt.target;
  if (lastTapNode === node) {
    clearTimeout(tapTimeout);
    lastTapNode = null;
    enterFocusMode(node);
  } else {
    lastTapNode = node;
    tapTimeout = setTimeout(() => { lastTapNode = null; }, 350);
  }
});

// --- Click background: reset ---
cy.on('tap', function(evt) {
  if (evt.target === cy) {
    if (focusMode) return;
    cy.elements().removeClass('highlighted faded');
    closePanel();
  }
});

// --- Keyboard: focus mode navigation ---
document.addEventListener('keydown', function(e) {
  if (!focusMode) return;
  if (e.key === 'Escape') {
    exitFocusMode();
  } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
    e.preventDefault();
    focusNeighborIdx = (focusNeighborIdx + 1) % focusNeighbors.length;
    highlightFocusNeighbor();
  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
    e.preventDefault();
    focusNeighborIdx = (focusNeighborIdx - 1 + focusNeighbors.length) % focusNeighbors.length;
    highlightFocusNeighbor();
  } else if (e.key === 'Enter') {
    // Jump to selected neighbor
    if (focusNeighbors.length > 0) {
      const nb = focusNeighbors[focusNeighborIdx];
      const nbNode = cy.getElementById(nb.id);
      exitFocusMode();
      selectNode(nbNode);
      enterFocusMode(nbNode);
    }
  }
});

function selectNode(node) {
  const d = node.data();

  // Highlight neighborhood
  cy.elements().removeClass('highlighted faded trail-node trail-edge');
  const neighborhood = node.closedNeighborhood();
  neighborhood.addClass('highlighted');
  cy.elements().not(neighborhood).addClass('faded');

  // Re-apply trail styling to trail nodes/edges
  trail.forEach((t, i) => {
    const trailNode = cy.getElementById(t.id);
    trailNode.removeClass('faded').addClass('trail-node');
    if (i > 0) {
      const prevId = trail[i - 1].id;
      cy.getElementById(prevId).edgesWith(trailNode).removeClass('faded').addClass('trail-edge');
    }
  });

  // Add to trail
  if (trail.length === 0 || trail[trail.length - 1].id !== d.id) {
    // Check if this node is a neighbor of the last trail node
    let relLabel = '';
    let isConnected = false;
    if (trail.length > 0) {
      const lastId = trail[trail.length - 1].id;
      const edges = cy.getElementById(lastId).edgesWith(node);
      if (edges.length > 0) {
        relLabel = edges[0].data('label') || '';
        isConnected = true;
      }
    }

    // If not connected to last trail node, reset trail
    if (trail.length > 0 && !isConnected) {
      cy.elements().removeClass('trail-node trail-edge');
      trail.length = 0;
    }

    trail.push({ id: d.id, label: d.label, relationLabel: relLabel });
    node.addClass('trail-node');
    if (trail.length > 1) {
      const prevId = trail[trail.length - 2].id;
      cy.getElementById(prevId).edgesWith(node).addClass('trail-edge');
    }
    renderTrail();
  }

  showPanel(node);
}

function enterFocusMode(node) {
  focusMode = true;
  focusNode = node;
  const neighbors = node.neighborhood('node');
  focusNeighbors = neighbors.map(n => ({ id: n.id(), label: n.data('label') }));
  focusNeighborIdx = 0;

  // Fade everything, show only neighborhood
  cy.elements().removeClass('highlighted faded');
  const neighborhood = node.closedNeighborhood();
  neighborhood.addClass('highlighted');
  cy.elements().not(neighborhood).addClass('faded');

  // Zoom to neighborhood
  cy.animate({ fit: { eles: neighborhood, padding: 80 } }, { duration: 400 });

  document.getElementById('focus-hint').classList.add('visible');
  if (focusNeighbors.length > 0) highlightFocusNeighbor();
}

function exitFocusMode() {
  focusMode = false;
  focusNode = null;
  focusNeighbors = [];
  cy.elements().removeClass('highlighted faded');
  cy.animate({ fit: { padding: 80 } }, { duration: 400 });
  document.getElementById('focus-hint').classList.remove('visible');
}

function highlightFocusNeighbor() {
  if (focusNeighbors.length === 0) return;
  const nb = focusNeighbors[focusNeighborIdx];
  // Reset all neighbor highlights, then highlight current
  focusNeighbors.forEach(n => {
    const el = cy.getElementById(n.id);
    el.removeClass('hover');
  });
  const current = cy.getElementById(nb.id);
  current.addClass('hover');
  showPanel(current);
}

function showPanel(node) {
  const d = node.data();
  const panel = document.getElementById('panel');
  const content = document.getElementById('panel-content');

  let obsHtml = '';
  if (d.observations && d.observations.length > 0) {
    obsHtml = '<div class="section"><h3>Observations (' + d.observations.length + ')</h3>';
    d.observations.forEach(o => {
      obsHtml += '<div class="obs">' + escapeHtml(o.content);
      if (o.source) obsHtml += '<div class="source">' + escapeHtml(o.source) + '</div>';
      obsHtml += '</div>';
    });
    obsHtml += '</div>';
  }

  let relHtml = '';
  if (d.relations && d.relations.length > 0) {
    relHtml = '<div class="section"><h3>Relations (' + d.relations.length + ')</h3>';
    d.relations.forEach(r => {
      relHtml += '<div class="rel">' + escapeHtml(r) + '</div>';
    });
    relHtml += '</div>';
  }

  content.innerHTML =
    '<h2>' + escapeHtml(d.label) + '</h2>' +
    '<span class="type-badge" style="background:' + d.color + '">' + escapeHtml(d.entity_type) + '</span>' +
    '<div class="entity-id">' + d.id + '</div>' +
    obsHtml + relHtml;

  panel.classList.add('open');
}

function closePanel() {
  document.getElementById('panel').classList.remove('open');
}

// --- Trail ---
function renderTrail() {
  const el = document.getElementById('trail');
  if (trail.length === 0) {
    el.classList.remove('visible');
    return;
  }
  el.classList.add('visible');
  let html = '';
  trail.forEach((t, i) => {
    if (i > 0) {
      if (t.relationLabel) {
        html += '<span class="relation-label">' + escapeHtml(t.relationLabel) + '</span>';
      }
      html += '<span class="arrow">→</span>';
    }
    html += '<span class="crumb" onclick="jumpToTrail(' + i + ')">' + escapeHtml(t.label) + '</span>';
  });
  html += '<button class="clear-btn" onclick="clearTrail()">clear</button>';
  el.innerHTML = html;
}

function jumpToTrail(idx) {
  const t = trail[idx];
  const node = cy.getElementById(t.id);
  if (node) {
    selectNode(node);
    cy.animate({ center: { eles: node }, zoom: cy.zoom() }, { duration: 300 });
  }
}

function clearTrail() {
  cy.elements().removeClass('trail-node trail-edge');
  trail.length = 0;
  renderTrail();
}

// --- Filters ---
function toggleTypeFilter(type, btn) {
  if (activeTypes.has(type)) {
    activeTypes.delete(type);
    btn.classList.remove('active');
  } else {
    activeTypes.add(type);
    btn.classList.add('active');
  }
  applyFilters();
}

function toggleRelFilter(type, btn) {
  if (activeRels.has(type)) {
    activeRels.delete(type);
    btn.classList.remove('active');
  } else {
    activeRels.add(type);
    btn.classList.add('active');
  }
  applyFilters();
}

function applyFilters() {
  cy.batch(() => {
    // Nodes
    cy.nodes().forEach(node => {
      if (activeTypes.has(node.data('entity_type'))) {
        node.removeClass('hidden');
      } else {
        node.addClass('hidden');
      }
    });
    // Edges
    cy.edges().forEach(edge => {
      const relType = edge.data('label');
      const srcVisible = !edge.source().hasClass('hidden');
      const tgtVisible = !edge.target().hasClass('hidden');
      if (activeRels.has(relType) && srcVisible && tgtVisible) {
        edge.removeClass('hidden');
      } else {
        edge.addClass('hidden');
      }
    });
  });
  drawHulls();
}

// --- Community hulls ---
const canvas = document.getElementById('hull-canvas');
const ctx = canvas.getContext('2d');
let showHulls = false;

document.getElementById('hull-toggle').addEventListener('change', function() {
  showHulls = this.checked;
  drawHulls();
});

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  drawHulls();
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

cy.on('render', drawHulls);
cy.on('pan zoom', drawHulls);

function drawHulls() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!showHulls || !graphData.communities) return;

  graphData.communities.forEach((community, i) => {
    if (community.length < 2) return;

    // Get rendered positions of community nodes (skip hidden)
    const points = [];
    community.forEach(nodeId => {
      const node = cy.getElementById(nodeId);
      if (node && node.length > 0 && !node.hasClass('hidden') && !node.hasClass('faded')) {
        const pos = node.renderedPosition();
        const size = node.renderedWidth() / 2;
        points.push({ x: pos.x, y: pos.y, r: size });
      }
    });

    if (points.length < 2) return;

    const colorIdx = i % graphData.communityColors.length;
    const fillColor = graphData.communityColors[colorIdx];
    const borderColor = graphData.communityBorderColors[colorIdx];

    // Draw expanded convex hull
    const hull = convexHull(points);
    if (hull.length < 2) return;

    const padding = 25;
    ctx.beginPath();
    // Draw rounded hull using arc segments
    for (let j = 0; j < hull.length; j++) {
      const p0 = hull[(j - 1 + hull.length) % hull.length];
      const p1 = hull[j];
      const p2 = hull[(j + 1) % hull.length];

      // Offset outward
      const dx1 = p1.x - p0.x, dy1 = p1.y - p0.y;
      const dx2 = p2.x - p1.x, dy2 = p2.y - p1.y;
      const nx = -(dy1 + dy2), ny = (dx1 + dx2);
      const len = Math.sqrt(nx * nx + ny * ny) || 1;

      const ox = p1.x + (nx / len) * padding;
      const oy = p1.y + (ny / len) * padding;

      if (j === 0) ctx.moveTo(ox, oy);
      else ctx.lineTo(ox, oy);
    }
    ctx.closePath();

    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function convexHull(points) {
  if (points.length < 2) return points;
  const pts = points.map(p => [p.x, p.y]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  const cross = (O, A, B) => (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);

  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (const p of pts.reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  upper.pop(); lower.pop();
  return lower.concat(upper).map(p => ({ x: p[0], y: p[1] }));
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
</script>
</body>
</html>"""


def tool_visualize_graph(vault: str = "") -> str:
    """Generate an interactive knowledge graph visualization and open in browser.

    Args:
        vault: Optional vault filter (empty = all vaults).

    Returns:
        Status message with file path.
    """
    graph = get_graph()

    # Collect all entities from the store
    entities_list, total = list_entities(
        vault=vault or None, entity_type=None, offset=0, limit=9999
    )
    entity_map = {e.id: e for e in entities_list}

    # Detect communities
    communities = detect_communities()
    # Map entity_id -> community_index
    entity_community = {}
    for i, community in enumerate(communities):
        for eid in community:
            entity_community[eid] = i

    # Build nodes
    nodes = []
    for entity in entities_list:
        obs_list = get_observations(entity.id, include_superseded=False)
        relations = []

        for _, target, data in graph.out_edges(entity.id, data=True):
            target_ent = entity_map.get(target) or get_entity(target)
            target_name = target_ent.name if target_ent else target[:8]
            relations.append(f"\u2192 {target_name} [{data.get('relation_type', '?')}]")
        for source, _, data in graph.in_edges(entity.id, data=True):
            source_ent = entity_map.get(source) or get_entity(source)
            source_name = source_ent.name if source_ent else source[:8]
            relations.append(f"\u2190 {source_name} [{data.get('relation_type', '?')}]")

        degree = graph.degree(entity.id) if graph.has_node(entity.id) else 0
        size = max(16, min(55, 16 + degree * 4))
        font_size = max(8, min(14, 8 + degree * 1))

        color = TYPE_COLORS.get(entity.entity_type, DEFAULT_COLOR)

        nodes.append({
            "data": {
                "id": entity.id,
                "label": entity.name,
                "entity_type": entity.entity_type,
                "color": color,
                "size": size,
                "fontSize": font_size,
                "community": entity_community.get(entity.id, -1),
                "observations": [
                    {"content": o.content, "source": o.source}
                    for o in obs_list
                ],
                "relations": relations,
            }
        })

    # Build edges
    edges = []
    all_relations = get_all_relations()
    for rel in all_relations:
        if vault:
            from_ent = entity_map.get(rel.from_entity)
            to_ent = entity_map.get(rel.to_entity)
            if not from_ent or not to_ent:
                continue

        edges.append({
            "data": {
                "id": rel.id,
                "source": rel.from_entity,
                "target": rel.to_entity,
                "label": rel.relation_type,
            }
        })

    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "typeColors": TYPE_COLORS,
        "defaultColor": DEFAULT_COLOR,
        "communities": communities,
        "communityColors": COMMUNITY_COLORS,
        "communityBorderColors": COMMUNITY_BORDER_COLORS,
    }

    # Generate HTML
    html = HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(graph_data))

    # Write to temp file and open
    out_dir = Path(tempfile.gettempdir()) / "memory-index"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "graph.html"
    out_file.write_text(html, encoding="utf-8")

    # Open in browser
    try:
        if sys.platform == "win32":
            os.startfile(str(out_file))
        else:
            webbrowser.open(out_file.as_uri())
    except Exception as e:
        logger.warning("Could not open browser: %s", e)

    return f"Graph visualization opened in browser.\nFile: {out_file}\n{len(nodes)} entities, {len(edges)} relations."
