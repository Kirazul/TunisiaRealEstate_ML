const GEOJSON_PATH = "assets/data/atlas.geojson";
const COVERAGE_PATH = "assets/data/zone_coverage.json";
const SUMMARY_PATH = "/model_summary";
const canvas = document.getElementById("atlas-canvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const landing = document.getElementById("hud-landing");
const detail = document.getElementById("hud-detail");
const aiPanel = document.getElementById("hud-ai-info");
const aiText = document.getElementById("ai-text");
const cineOverlay = document.getElementById("cinematic-overlay");
const fpsElem = document.getElementById("fps-counter");
const zoneModal = document.getElementById("zone-modal");
const zoneModalMessage = document.getElementById("zone-modal-message");
const zoneModalClose = document.getElementById("zone-modal-close");
const familySelect = document.getElementById("family-select");
const surfaceSlider = document.getElementById("surface-slider");
const surfaceInput = document.getElementById("surface-input");
const surfaceSliderValue = document.getElementById("surface-slider-value");

const projection = d3.geoMercator();
const geoPath = d3.geoPath().projection(projection);
const spatialIndex = new RBush();
const zoom = d3.zoom().scaleExtent([1, 40]).on("zoom", (event) => {
    transform = event.transform;
    draw();
});

let transform = d3.zoomIdentity;
let features = [];
let paths = [];
let hovered = null;
let selected = null;
let atlasReady = false;
let offCanvas = null;
let offCtx = null;
let coverageMap = new Map();
let summary = null;

let lastFrameTime = performance.now();
let frameCount = 0;
let predictionRefreshTimer = null;
let activeMapMode = "support";
let activeFamilyOverride = null;

function getRegionCode(feature) {
    return String(feature.properties.region_code || feature.properties.delegation_key || "");
}

function sanitizeJsonArray(data) {
    return Array.isArray(data) ? data : [];
}

function normalizeCoverageRecord(record) {
    return {
        ...record,
        region_code: String(record.region_code || record.delegation_key || ""),
        has_enough_data: Boolean(record.has_enough_data),
        support_count: Number(record.support_count || 0),
        prediction: typeof record.prediction === "number" ? record.prediction : null,
    };
}

function isRenderableFeature(feature) {
    const bounds = d3.geoBounds(feature);
    return bounds[0][0] >= 6 && bounds[1][0] <= 13 && bounds[0][1] >= 30 && bounds[1][1] <= 38;
}

function getCoverage(feature) {
    return coverageMap.get(getRegionCode(feature)) || null;
}

function getCoverageProfiles(coverage) {
    return coverage && coverage.profiles && typeof coverage.profiles === "object"
        ? Object.values(coverage.profiles)
        : [];
}

function hasFamilyDirectCoverage(coverage) {
    return getCoverageProfiles(coverage).some((profile) => profile?.coverage_level === "exact_sector");
}

function getSummaryModelName() {
    return summary?.best_model || summary?.model_name || "V2 model";
}

function getSummaryAccuracyPct() {
    if (typeof summary?.accuracy_pct === "number") return summary.accuracy_pct;
    if (typeof summary?.validation_r2 === "number") return summary.validation_r2 * 100;
    const modelResult = Array.isArray(summary?.model_results) ? summary.model_results[0] : null;
    if (typeof modelResult?.r2 === "number") return modelResult.r2 * 100;
    return 0;
}

function getTierColor(level) {
    if (activeMapMode === "exact") {
        return level === "exact_sector" ? "#ffffff" : "rgba(255,255,255,0.06)";
    }
    return {
        exact_sector: "#ff6b6b",
        locality_fallback: "#ffffff",
        delegation_fallback: "#ffffff",
        governorate_fallback: "#ffffff",
        national_fallback: "#ffffff",
    }[level] || "rgba(255,255,255,0.22)";
}

function getRegionColor(feature) {
    const coverage = getCoverage(feature);
    if (!coverage || !coverage.has_enough_data || coverage.prediction === null || coverage.prediction === undefined) {
        return "#1a1a1d";
    }
    if (activeMapMode === "support") {
        return hasFamilyDirectCoverage(coverage) ? "#ef4444" : "#7f1d1d";
    }
    return hasFamilyDirectCoverage(coverage) ? "#ff4d4f" : "rgba(70, 12, 12, 0.55)";
}

function formatCoverageLabel(level) {
    const map = {
        exact_sector: "Exact sector",
        locality_fallback: "Locality fallback",
        delegation_fallback: "Delegation fallback",
        governorate_fallback: "Governorate fallback",
        national_fallback: "National fallback"
    };
    return map[level] || String(level || "unknown").replace(/_/g, " ");
}

function getSelectedFamily(coverage) {
    if (familySelect) return familySelect.value;
    return activeFamilyOverride || coverage?.default_family || "apartment";
}

function getActiveProfile(coverage) {
    if (!coverage) return null;
    const selectedFamily = getSelectedFamily(coverage);
    return coverage.profiles?.[selectedFamily] || coverage.profiles?.[coverage.default_family || "apartment"] || null;
}

function formatCurrency(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "--";
    return Math.round(value).toLocaleString();
}

function getFamilyLabel(family) {
    return family === "apartment" ? "Apartment" : family === "house" ? "House" : "Land";
}

function buildScenarioFromProfile(coverage) {
    if (!coverage || !coverage.has_enough_data) return null;
    const family = getSelectedFamily(coverage);
    const profile = coverage.profiles?.[family] || coverage.profiles?.[coverage.default_family];
    if (!profile) return null;
    const surface = Number(surfaceSlider ? surfaceSlider.value : profile.surface);
    const pricePerM2 = Number(profile.price_per_m2 || 0);
    return {
        family,
        familyLabel: getFamilyLabel(family),
        surface,
        price_per_m2: pricePerM2,
        prediction: pricePerM2 * surface,
        property_type: family,
        nature: "sale",
        coverage_level: profile.coverage_level,
        support_count: Number(profile.support_count || coverage.support_count || 0),
        base_surface: Number(profile.surface || 0),
    };
}

function getFeatureDelegation(feature) {
    return feature.properties.name_fr || feature.properties.NomDelegat || feature.properties.delegation || "Delegation";
}

function getFeatureName(feature) {
    return feature.properties.name_fr || feature.properties.delegation || "Zone";
}

function renderFamilyProfiles(profiles) {
    const container = document.getElementById("family-profiles");
    if (!container) return;
    container.innerHTML = "";
    
    // We determine the active family from our internal state or the hidden dropdown (if it still exists in DOM)
    // To be safe, we'll check the 'active' state of cards in the next step.
    const activeFamily = selected ? getSelectedFamily(getCoverage(selected)) : (activeFamilyOverride || "apartment");
    
    ["apartment", "house", "land"].forEach((familyName) => {
        const profile = profiles ? profiles[familyName] : null;
        if (!profile) return;
        const card = document.createElement("div");
        card.className = `family-card${activeFamily === familyName ? " active" : ""}`;
        card.innerHTML = `
            <h4>${getFamilyLabel(familyName)}</h4>
            <div class="family-main">${Number(profile.price_per_m2).toLocaleString()} TND/m²</div>
            <div class="family-meta">
                Type : ${getFamilyLabel(profile.property_type)}<br>
                Nature : Sale<br>
                Base surface : ${Math.round(profile.surface)} m²<br>
                Level : <span class="tier-badge tier-${profile.coverage_level}">${formatCoverageLabel(profile.coverage_level)}</span>
            </div>
        `;
        card.addEventListener("click", () => {
            if (familySelect) familySelect.value = familyName;
            activeFamilyOverride = familyName;
            if (selected) {
                // Instantly update everything when switching family
                refreshDetailView(selected);
                loadPredictionForFeature(selected);
            }
        });
        container.appendChild(card);
    });
}

function renderSelectedProfile(selectedProfile) {
    const container = document.getElementById("selected-profile");
    if (!container) return;
    if (!selectedProfile) {
        container.classList.add("hidden");
        container.innerHTML = "";
        return;
    }
    container.classList.remove("hidden");
    container.innerHTML = `
        <strong>${selectedProfile.familyLabel} scenario</strong>
        <span>Benchmark price : ${formatCurrency(selectedProfile.price_per_m2)} TND/m²</span>
        <span>Selected surface : ${formatCurrency(selectedProfile.surface)} m²</span>
        <span>Total formula : ${formatCurrency(selectedProfile.price_per_m2)} × ${formatCurrency(selectedProfile.surface)} = ${formatCurrency(selectedProfile.prediction)} TND</span>
        <span>Support : ${formatCurrency(selectedProfile.support_count)} observations · Tier : ${formatCoverageLabel(selectedProfile.coverage_level)}</span>
    `;
}

function syncInteractiveControls(coverage, preserveFamily = false) {
    if (!coverage) return;
    const defaultFamily = coverage.default_family || "apartment";
    if (familySelect && !preserveFamily) familySelect.value = defaultFamily;
    if (!preserveFamily) activeFamilyOverride = defaultFamily;
    const profiles = coverage.profiles || {};
    const activeFamily = getSelectedFamily(coverage);
    const defaultProfile = profiles[activeFamily] || profiles[defaultFamily] || Object.values(profiles)[0];
    if (defaultProfile) {
        const suggestedSurface = Math.max(Number(surfaceSlider.min), Math.min(Number(surfaceSlider.max), Math.round(defaultProfile.surface)));
        setSurfaceValue(suggestedSurface);
    }
}

function setSurfaceValue(value) {
    const numeric = Math.max(1, Math.min(5000, Math.round(Number(value) || 1)));
    if (surfaceSlider) surfaceSlider.value = numeric;
    if (surfaceInput) surfaceInput.value = numeric;
    if (surfaceSliderValue) surfaceSliderValue.textContent = `${numeric.toLocaleString()} m²`;
    return numeric;
}

function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (atlasReady) {
        rebakeAtlas();
        draw();
    }
}

function rebakeAtlas() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const dpr = window.devicePixelRatio || 1;
    if (!offCanvas) offCanvas = document.createElement("canvas");
    offCanvas.width = width * dpr;
    offCanvas.height = height * dpr;
    offCtx = offCanvas.getContext("2d");
    offCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    offCtx.clearRect(0, 0, width, height);
    paths.forEach((path, index) => {
        const feature = features[index];
        const coverage = getCoverage(feature);
        offCtx.fillStyle = getRegionColor(feature);
        offCtx.fill(path);
        offCtx.strokeStyle = getTierColor(coverage?.coverage_level);
        offCtx.lineWidth = coverage ? 0.9 : 0.45;
        offCtx.stroke(path);
    });
}

function draw() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    ctx.clearRect(0, 0, width, height);
    if (!atlasReady || !offCanvas) return;

    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    const isHighZoom = transform.k > 1.25;

    if (!selected) {
        if (isHighZoom) {
            // HD VECTOR PASSTHROUGH
            // When zooming, we redraw the paths directly for perfect clarity
            paths.forEach((path, index) => {
                const feature = features[index];
                const coverage = getCoverage(feature);
                ctx.fillStyle = getRegionColor(feature);
                ctx.fill(path);
                ctx.strokeStyle = getTierColor(coverage?.coverage_level);
                ctx.lineWidth = (coverage ? 0.9 : 0.45) / transform.k;
                ctx.stroke(path);
            });
        } else {
            // Performance mode: draw pre-baked bitmap
            ctx.drawImage(offCanvas, 0, 0, width, height);
        }
    }

    const hoverStroke = 1.4 / transform.k;
    const selectedStroke = 3.0 / transform.k;

    if (hovered) {
        const path = hovered.__path;
        ctx.fillStyle = "rgba(255,255,255,0.08)";
        ctx.fill(path);
        ctx.strokeStyle = "rgba(255,255,255,0.6)";
        ctx.lineWidth = hoverStroke;
        ctx.stroke(path);
    }

    if (selected) {
        const path = selected.__path;
        // Solo-Mode: The atlas is already hidden in !selected block
        // We draw the target with high-fidelity Red accent
        ctx.fillStyle = "rgba(255,45,85,0.12)";
        ctx.fill(path);
        ctx.strokeStyle = "#ff2d55"; // VIBRANT RED SELECTION
        ctx.lineWidth = selectedStroke;
        ctx.stroke(path);

        // Sub-stroke for premium "glow" effect
        ctx.strokeStyle = "rgba(255,45,85,0.3)";
        ctx.lineWidth = selectedStroke * 2.5;
        ctx.stroke(path);
    }
    ctx.restore();
}

function hitTest(clientX, clientY) {
    const x = (clientX - transform.x) / transform.k;
    const y = (clientY - transform.y) / transform.k;
    const candidates = spatialIndex.search({ minX: x, minY: y, maxX: x, maxY: y });
    for (let index = candidates.length - 1; index >= 0; index -= 1) {
        const item = candidates[index];
        if (ctx.isPointInPath(item.path, x, y)) return item.feature;
    }
    return null;
}

function isPointerInsideFeature(feature, clientX, clientY) {
    if (!feature || !feature.__path) return false;
    const x = (clientX - transform.x) / transform.k;
    const y = (clientY - transform.y) / transform.k;
    return ctx.isPointInPath(feature.__path, x, y);
}

function updateTooltip(feature, clientX, clientY) {
    if (!feature) {
        tooltip.style.display = "none";
        return;
    }
    const coverage = getCoverage(feature);
    const profile = getActiveProfile(coverage);
    const delegation = getFeatureDelegation(feature);
    const governorate = coverage?.governorate || feature.properties.governorate || "Tunisie";
    let body = `<div class="tooltip-location"><span>${delegation}</span><small>${governorate}</small></div>`;
    if (!coverage || !coverage.has_enough_data) {
        body += `<div class="tooltip-meta tooltip-meta-alert">Insufficient data</div>`;
    } else {
        body += `<div class="tooltip-metrics">`;
        body += `<div><label>Reference</label><strong>${Number(profile?.price_per_m2 ?? coverage.prediction).toLocaleString()} TND/m²</strong></div>`;
        body += `<div><label>Support</label><strong>${Number(profile?.support_count ?? coverage.support_count ?? 0).toLocaleString()} obs</strong></div>`;
        body += `<div><label>Tier</label><strong>${formatCoverageLabel(profile?.coverage_level || coverage.coverage_level)}</strong></div>`;
        body += `</div>`;
    }
    tooltip.innerHTML = `<div class="tooltip-kicker">${getFeatureName(feature)}</div>${body}`;
    tooltip.style.left = `${clientX + 16}px`;
    tooltip.style.top = `${clientY - 12}px`;
    tooltip.style.display = "block";
}

function applySummaryStats() {
    if (!summary) return;
    const { totalDelegations, atlasCovered, directCovered, fallbackCovered } = getCoverageCounts();
    document.getElementById("stat-accuracy").textContent = `${getSummaryAccuracyPct().toFixed(1)}%`;
    document.getElementById("stat-direct").textContent = `${directCovered.toLocaleString()}/${totalDelegations.toLocaleString()}`;
    document.getElementById("stat-assisted").textContent = `${fallbackCovered.toLocaleString()}/${totalDelegations.toLocaleString()}`;
    renderLegendContent();
}

function openZoneModal(message) {
    zoneModalMessage.textContent = message;
    zoneModal.classList.remove("hidden");
}

function closeZoneModal() {
    zoneModal.classList.add("hidden");
}

function generateModelNarrative(feature) {
    const coverage = getCoverage(feature);
    const selectedFamily = getSelectedFamily(coverage);
    const profile = getActiveProfile(coverage);
    const governorate = coverage?.governorate || feature.properties.governorate || "Tunisie";
    const delegation = getFeatureDelegation(feature);
    const locality = getFeatureName(feature);
    const labels = { apartment: "Apartment", house: "House", land: "Land" };
    const enFamily = labels[selectedFamily] || selectedFamily;

    if (!coverage || !coverage.has_enough_data) {
        aiText.textContent = `${locality} in ${delegation}, ${governorate} is currently outside the reliable support envelope of the trained model.`;
        return;
    }
    aiText.textContent = `${locality} in ${delegation}, ${governorate} is currently analyzed via the ${enFamily} market profile. This selected family operates at the ${formatCoverageLabel(profile?.coverage_level || coverage.coverage_level)} tier with ${Number(profile?.support_count || coverage.support_count || 0).toLocaleString()} supporting observations. At the delegation level, this atlas currently classifies the zone as ${formatCoverageLabel(coverage.coverage_level)}. The deployed ${getSummaryModelName()} model achieved a validation R² of ${getSummaryAccuracyPct().toFixed(2)}%, and the selected-family benchmark is ${Number(profile?.price_per_m2 || coverage.prediction).toLocaleString()} TND per square meter.`;
}

function refreshDetailView(feature) {
    const coverage = getCoverage(feature);
    const governorate = coverage?.governorate || feature.properties.governorate || "Tunisia";
    const delegation = getFeatureDelegation(feature);
    const profile = getActiveProfile(coverage);
    document.getElementById("lbl-sector").textContent = getFeatureName(feature);
    document.getElementById("lbl-deleg").textContent = delegation;
    document.getElementById("lbl-gouv").textContent = `${governorate}, Tunisia`;
    document.getElementById("bc-gov").textContent = governorate;
    document.getElementById("bc-deleg").textContent = delegation;
    document.getElementById("bc-sector").textContent = getFeatureName(feature);
    document.getElementById("meta-coverage").textContent = coverage && coverage.has_enough_data ? formatCoverageLabel(coverage.coverage_level) : "Missing";
    document.getElementById("meta-model").textContent = summary ? getSummaryModelName() : "--";
    syncInteractiveControls(coverage, true);
    renderFamilyProfiles(coverage?.profiles || {});
    generateModelNarrative(feature);
}

function getBestModelMetrics() {
    if (!summary || !Array.isArray(summary.model_results)) return { r2: null, rmse: null };
    const found = summary.model_results.find((item) => item.model === getSummaryModelName()) || summary.model_results[0];
    return { r2: found?.r2 ?? null, rmse: found?.rmse ?? null };
}

function getCoverageCounts() {
    const records = Array.from(coverageMap.values());
    const totalDelegations = Number(summary?.total_delegations || features.length || records.length || 0);
    const atlasCovered = records.filter((coverage) => coverage?.has_enough_data).length || Number(summary?.covered_delegations || totalDelegations || 0);
    const directCovered = records.filter((coverage) => coverage?.has_enough_data && hasFamilyDirectCoverage(coverage)).length;
    return {
        totalDelegations,
        atlasCovered,
        directCovered,
        fallbackCovered: Math.max(0, atlasCovered - directCovered),
    };
}

function renderLegendContent() {
    const fallbackLegend = document.getElementById("fallback-legend");
    const legendCaption = document.getElementById("legend-caption");
    const legendLine1 = document.getElementById("legend-line-1");
    const legendLine2 = document.getElementById("legend-line-2");
    if (!summary || !fallbackLegend || !legendCaption || !legendLine1 || !legendLine2) return;

    const { totalDelegations, atlasCovered, directCovered, fallbackCovered } = getCoverageCounts();

    legendLine1.textContent = activeMapMode === "exact"
        ? "Bright red highlights exact local support"
        : "Bright red marks exact local support";
    legendLine2.textContent = activeMapMode === "exact"
        ? "Shows only delegations with direct training data"
        : "Soft red marks delegations using fallback data";
    fallbackLegend.innerHTML = "";
    legendCaption.textContent = "Bright red marks delegations with direct training data. Soft red areas use fallback benchmarks from broader regions.";
}

function setMapMode(mode) {
    activeMapMode = mode;
    document.querySelectorAll(".map-mode-btn").forEach((button) => {
        button.classList.toggle("active", button.dataset.mode === mode);
    });
    renderLegendContent();
    if (atlasReady) {
        rebakeAtlas();
        draw();
    }
}

function getConfidenceFromProfile(profile) {
    const level = profile.coverage_level;
    if (level === "exact_sector") return "HIGH";
    if (level === "locality_fallback") return "MEDIUM";
    if (level === "delegation_fallback") return "MEDIUM";
    if (level === "governorate_fallback") return "MEDIUM";
    return "LOW";
}

async function loadPredictionForFeature(feature) {
    const coverage = getCoverage(feature);
    const resultBox = document.getElementById("result");
    const resultMessage = document.getElementById("result-message");
    if (!coverage || !coverage.has_enough_data) {
        resultBox.classList.add("hidden");
        openZoneModal("This zone does not have sufficient support to produce a reliable prediction with the production model.");
        return;
    }
    const scenario = buildScenarioFromProfile(coverage);
    resultBox.classList.remove("hidden");
    document.getElementById("result-label").textContent = "SELECTED SCENARIO PRICE";
    document.getElementById("res-price").textContent = "...";
    document.getElementById("res-unit").textContent = "TND";
    document.getElementById("res-accuracy").textContent = "...";
    renderFamilyProfiles(coverage.profiles || {});
    resultMessage.textContent = "Computing scenario from the V2 benchmark...";
    try {
        if (!scenario) {
            throw new Error("Scenario benchmark unavailable");
        }
        document.getElementById("res-price").textContent = formatCurrency(scenario.prediction);
        document.getElementById("res-unit").textContent = "TND";
        document.getElementById("result-label").textContent = `TOTAL SCENARIO ${scenario.familyLabel.toUpperCase()}`;
        document.getElementById("res-accuracy").textContent = String(getConfidenceFromProfile({ coverage_level: scenario.coverage_level })).toUpperCase();
        renderFamilyProfiles(coverage.profiles || {});
        renderSelectedProfile(scenario);
        resultMessage.textContent = `V2 benchmark mode: total price is computed directly from the selected family benchmark in this delegation. ${formatCurrency(scenario.price_per_m2)} TND/m² multiplied by ${formatCurrency(scenario.surface)} m² gives ${formatCurrency(scenario.prediction)} TND.`;
    } catch (error) {
        console.error(error);
        renderSelectedProfile(null);
        resultMessage.textContent = "Scenario computation failed. Check V2 coverage data.";
    }
}

function backToAtlas() {
    selected = null;
    hovered = null;
    detail.classList.remove("active");
    aiPanel.classList.remove("active");
    if (cineOverlay) cineOverlay.classList.remove("active");
    setTimeout(() => landing.classList.add("active"), 200);
    d3.select(canvas).transition().duration(800).ease(d3.easeCubicInOut).call(zoom.transform, d3.zoomIdentity);
    tooltip.style.display = "none";
    draw();
}

function selectFeature(feature, clientX, clientY) {
    selected = feature;
    activeFamilyOverride = null;
    landing.classList.remove("active");
    detail.classList.add("active");
    aiPanel.classList.add("active");
    if (cineOverlay) {
        cineOverlay.style.setProperty("--cx", `${(clientX / window.innerWidth) * 100}%`);
        cineOverlay.style.setProperty("--cy", `${(clientY / window.innerHeight) * 100}%`);
        cineOverlay.classList.add("active");
    }
    const [[x0, y0], [x1, y1]] = geoPath.bounds(feature);
    const width = window.innerWidth;
    const height = window.innerHeight;
    const scale = Math.min(20, 0.82 / Math.max((x1 - x0) / width, (y1 - y0) / height));
    const translateX = width / 2 - scale * (x0 + x1) / 2;
    const translateY = height / 2 - scale * (y0 + y1) / 2;
    d3.select(canvas).transition().duration(900).ease(d3.easeCubicInOut).call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
    syncInteractiveControls(getCoverage(feature), false);
    refreshDetailView(feature);
    loadPredictionForFeature(feature);
    draw();
}

function initSearch() {
    const input = document.getElementById("search-input");
    const results = document.getElementById("search-results");
    if (!input || !results) return;
    input.addEventListener("input", (event) => {
        const query = event.target.value.toLowerCase().trim();
        if (query.length < 2) {
            results.classList.add("hidden");
            results.innerHTML = "";
            return;
        }
        const matches = features.filter((feature) => {
            const sector = getFeatureName(feature).toLowerCase();
            const delegation = getFeatureDelegation(feature).toLowerCase();
            const gov = (getCoverage(feature)?.governorate || feature.properties.governorate || "").toLowerCase();
            return sector.includes(query) || delegation.includes(query) || gov.includes(query);
        }).slice(0, 12);
        results.innerHTML = "";
        matches.forEach((feature) => {
            const coverage = getCoverage(feature);
            const item = document.createElement("div");
            item.className = "search-item";
            item.innerHTML = `
                <span class="name">${getFeatureName(feature)}</span>
                <span class="meta">${getFeatureDelegation(feature)} · ${coverage ? formatCoverageLabel(coverage.coverage_level) : "missing data"}</span>
            `;
            item.addEventListener("mouseenter", () => { hovered = feature; draw(); });
            item.addEventListener("mouseleave", () => { hovered = null; draw(); });
            item.addEventListener("click", () => {
                results.classList.add("hidden");
                input.value = "";
                const [[x0, y0], [x1, y1]] = geoPath.bounds(feature);
                selectFeature(feature, (x0 + x1) / 2, (y0 + y1) / 2);
            });
            results.appendChild(item);
        });
        results.classList.toggle("hidden", matches.length === 0);
    });
    document.addEventListener("click", (event) => {
        if (!input.contains(event.target) && !results.contains(event.target)) results.classList.add("hidden");
    });
}

function monitorPerformance() {
    frameCount += 1;
    const now = performance.now();
    if (now - lastFrameTime > 500) {
        const fps = Math.round((frameCount * 1000) / (now - lastFrameTime));
        if (fpsElem) fpsElem.textContent = `${fps} FPS`;
        frameCount = 0;
        lastFrameTime = now;
    }
    requestAnimationFrame(monitorPerformance);
}

canvas.addEventListener("mousemove", (event) => {
    const hit = selected
        ? (isPointerInsideFeature(selected, event.clientX, event.clientY) ? selected : null)
        : hitTest(event.clientX, event.clientY);
    hovered = hit;
    canvas.style.cursor = hit ? "pointer" : "crosshair";
    updateTooltip(hit, event.clientX, event.clientY);
    draw();
});

canvas.addEventListener("mouseleave", () => {
    hovered = null;
    tooltip.style.display = "none";
    draw();
});

canvas.addEventListener("click", (event) => {
    if (selected) {
        backToAtlas();
        return;
    }
    const hit = hitTest(event.clientX, event.clientY);
    if (!hit) return;
    selectFeature(hit, event.clientX, event.clientY);
});

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && selected) {
        backToAtlas();
    }
});

if (zoneModalClose) {
    zoneModalClose.addEventListener("click", closeZoneModal);
}
if (zoneModal) {
    zoneModal.addEventListener("click", (event) => { if (event.target === zoneModal) closeZoneModal(); });
}

if (surfaceSlider && surfaceSliderValue) {
    surfaceSlider.addEventListener("input", () => {
        setSurfaceValue(surfaceSlider.value);
        if (selected) {
            clearTimeout(predictionRefreshTimer);
            predictionRefreshTimer = setTimeout(() => loadPredictionForFeature(selected), 120);
        }
    });
}

if (surfaceInput) {
    surfaceInput.addEventListener("input", () => { setSurfaceValue(surfaceInput.value); });
    surfaceInput.addEventListener("change", () => {
        setSurfaceValue(surfaceInput.value);
        if (selected) loadPredictionForFeature(selected);
    });
}

document.querySelectorAll(".preset-btn").forEach((button) => {
    button.addEventListener("click", () => {
        setSurfaceValue(button.getAttribute("data-surface"));
        if (selected) loadPredictionForFeature(selected);
    });
});

document.querySelectorAll(".map-mode-btn").forEach((button) => {
    button.addEventListener("click", () => {
        setMapMode(button.dataset.mode || "support");
    });
});

if (familySelect) {
    familySelect.addEventListener("change", () => {
        activeFamilyOverride = familySelect.value;
        if (selected) {
            refreshDetailView(selected);
            loadPredictionForFeature(selected);
        }
    });
}

setSurfaceValue(surfaceSlider ? surfaceSlider.value : 120);
resize();
window.addEventListener("resize", resize);
d3.select(canvas).call(zoom);
monitorPerformance();

Promise.all([
    d3.json(`${GEOJSON_PATH}?t=${Date.now()}`),
    d3.json(`${COVERAGE_PATH}?t=${Date.now()}`),
    d3.json(`${SUMMARY_PATH}?t=${Date.now()}`),
]).then(([geojson, coverageRecords, modelSummary]) => {
    summary = modelSummary;
    sanitizeJsonArray(coverageRecords).forEach((record) => {
        const normalized = normalizeCoverageRecord(record);
        coverageMap.set(normalized.region_code, normalized);
    });
    features = geojson.features.filter(isRenderableFeature);
    projection.fitExtent([[50, 50], [window.innerWidth - 50, window.innerHeight - 50]], { type: "FeatureCollection", features });
    const indexRows = [];
    paths = [];
    features.forEach((feature, index) => {
        const pathValue = geoPath(feature);
        if (!pathValue) return;
        const path = new Path2D(pathValue);
        feature.__path = path;
        paths[index] = path;
        const [[x0, y0], [x1, y1]] = geoPath.bounds(feature);
        indexRows.push({ minX: x0, minY: y0, maxX: x1, maxY: y1, feature, path });
    });
    spatialIndex.clear();
    spatialIndex.load(indexRows);
    applySummaryStats();
    initSearch();
    rebakeAtlas();
    atlasReady = true;
    draw();
    const loader = document.getElementById("loading-overlay");
    if (loader) loader.classList.add("hidden");
}).catch((error) => {
    console.error(error);
    const loader = document.getElementById("loading-overlay");
    if (loader) loader.querySelector("p").textContent = "ERREUR DE DONNÉES - VÉRIFIER LA CONSOLE";
});
