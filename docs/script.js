"use strict";

document.documentElement.classList.add("js");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function initHeroMotion() {
  const hero = document.querySelector(".hero");
  const visual = document.querySelector("[data-transport-visual]");
  if (!hero || !visual) return;

  document.documentElement.classList.add("hero-motion-enabled");

  let started = false;
  let transportInViewport = false;
  let hasSeenTransport = false;
  const restartTransport = () => {
    if (reducedMotion.matches) return;
    visual.classList.remove("is-running");
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => visual.classList.add("is-running"));
    });
  };

  const start = () => {
    if (!started) {
      started = true;
      hero.classList.add("is-animated");
    }
    restartTransport();
  };

  window.setTimeout(() => {
    if (!started) start();
  }, 900);

  const heroRect = hero.getBoundingClientRect();
  if (heroRect.bottom > 0 && heroRect.top < window.innerHeight * 0.8) {
    window.requestAnimationFrame(start);
  } else if ("IntersectionObserver" in window) {
    const heroObserver = new IntersectionObserver(
      (entries) => {
        if (!entries[0]?.isIntersecting) return;
        start();
        heroObserver.disconnect();
      },
      { threshold: 0.15 },
    );
    heroObserver.observe(hero);
  } else {
    start();
  }

  if ("IntersectionObserver" in window) {
    const pauseObserver = new IntersectionObserver(
      (entries) => {
        const isVisible = Boolean(entries[0]?.isIntersecting);
        const shouldReplay = isVisible && hasSeenTransport && !transportInViewport;

        transportInViewport = isVisible;
        visual.classList.toggle("is-paused", !isVisible || document.hidden);

        if (shouldReplay) restartTransport();
        if (isVisible) hasSeenTransport = true;
      },
      { threshold: 0.05 },
    );
    pauseObserver.observe(visual);
  } else {
    transportInViewport = true;
    hasSeenTransport = true;
  }

  document.addEventListener("visibilitychange", () => {
    visual.classList.toggle("is-paused", document.hidden || !transportInViewport);
    if (!document.hidden && transportInViewport && hasSeenTransport) restartTransport();
  });

  reducedMotion.addEventListener?.("change", (event) => {
    hero.classList.add("is-animated");
    if (event.matches) visual.classList.remove("is-running");
    else restartTransport();
  });
}

function initHeader() {
  const header = document.querySelector("[data-header]");
  if (!header) return;

  const update = () => header.classList.toggle("is-scrolled", window.scrollY > 18);
  update();
  window.addEventListener("scroll", update, { passive: true });
}

function initReveals() {
  const elements = [...document.querySelectorAll(".reveal")];
  if (!elements.length) return;

  if (reducedMotion.matches || !("IntersectionObserver" in window)) {
    elements.forEach((element) => element.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    {
      rootMargin: "0px 0px -8% 0px",
      threshold: 0.08,
    },
  );

  elements.forEach((element) => observer.observe(element));
}

function initSectionNavigation() {
  const links = [...document.querySelectorAll('.nav-links a[href^="#"]')];
  if (!links.length) return;

  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  const activate = (id) => {
    links.forEach((link) => {
      const current = link.getAttribute("href") === `#${id}`;
      if (current) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });
  };

  let framePending = false;
  const update = () => {
    const marker = window.scrollY + Math.min(window.innerHeight * 0.32, 280);
    let current = null;

    for (const section of sections) {
      if (section.offsetTop <= marker) current = section;
      else break;
    }

    activate(current?.id || null);
    framePending = false;
  };

  const requestUpdate = () => {
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(update);
  };

  update();
  window.addEventListener("scroll", requestUpdate, { passive: true });
  window.addEventListener("resize", requestUpdate, { passive: true });
}

function initSmartVideo() {
  const video = document.querySelector("[data-smart-video]");
  if (!video) return;

  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  const saveData = Boolean(connection && connection.saveData);
  const desktop = window.matchMedia("(min-width: 900px) and (hover: hover)").matches;
  const mayAutoplay = !reducedMotion.matches && !saveData && desktop;

  if (mayAutoplay && "IntersectionObserver" in window) {
    let attempted = false;
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (!entry || attempted || entry.intersectionRatio < 0.72) return;
        attempted = true;
        video.play().catch(() => {
        });
        observer.disconnect();
      },
      { threshold: [0.72] },
    );
    observer.observe(video);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && !video.paused) video.pause();
  });

  reducedMotion.addEventListener?.("change", (event) => {
    if (event.matches && !video.paused) video.pause();
  });
}

async function writeClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.appendChild(field);
  field.select();
  const copied = document.execCommand("copy");
  field.remove();
  if (!copied) throw new Error("Copy command was not accepted");
}

function initCitationCopy() {
  const button = document.querySelector("[data-copy-citation]");
  const citation = document.querySelector("#bibtex");
  const label = document.querySelector("[data-copy-label]");
  if (!button || !citation || !label) return;

  let resetTimer;
  button.addEventListener("click", async () => {
    window.clearTimeout(resetTimer);
    try {
      await writeClipboard(citation.textContent.trim());
      label.textContent = "Copied";
      button.setAttribute("aria-label", "BibTeX citation copied");
    } catch {
      label.textContent = "Select text";
      button.setAttribute("aria-label", "Copy failed; select the BibTeX text manually");
    }

    resetTimer = window.setTimeout(() => {
      label.textContent = "Copy";
      button.removeAttribute("aria-label");
    }, 2200);
  });
}

initHeroMotion();
initHeader();
initReveals();
initSectionNavigation();
initSmartVideo();
initCitationCopy();
