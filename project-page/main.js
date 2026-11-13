/* Project page behaviour: custom video player, figure lightbox, copy button. */
(function () {
  "use strict";

  /* --------------------------------------------------------------- player -- */
  var video = document.getElementById("concept-video");
  var player = document.getElementById("player");

  if (video && player) {
    var bigplay = document.getElementById("bigplay");
    var playpause = document.getElementById("playpause");
    var scrub = document.getElementById("scrub");
    var fill = document.getElementById("fill");
    var buffer = document.getElementById("buffer");
    var thumb = document.getElementById("thumb");
    var ticks = document.getElementById("ticks");
    var timeEl = document.getElementById("time");
    var speedBtn = document.getElementById("speed");
    var fsBtn = document.getElementById("fs");
    var chaptersEl = document.getElementById("chapters");

    var CHAPTERS = [
      { t: 0, label: "Intro" },
      { t: 5, label: "Problem" },
      { t: 18, label: "What fails" },
      { t: 23, label: "Gaussian: the fix" },
      { t: 39, label: "Gaussian: compare" },
      { t: 45, label: "Mixture: target" },
      { t: 54, label: "Mixture: flow lines" },
      { t: 59, label: "Mixture: learned flow" },
      { t: 71, label: "Same budget" },
      { t: 77, label: "Evidence" }
    ];
    var SPEEDS = [1, 1.5, 2, 0.5];
    var speedIdx = 0;
    var chapterBtns = [];

    function fmt(s) {
      if (!isFinite(s)) s = 0;
      var m = Math.floor(s / 60);
      var sec = Math.floor(s - m * 60);
      return m + ":" + (sec < 10 ? "0" : "") + sec;
    }

    function duration() {
      return isFinite(video.duration) ? video.duration : 90;
    }

    function buildChapters() {
      CHAPTERS.forEach(function (ch, i) {
        var b = document.createElement("button");
        b.type = "button";
        b.textContent = fmt(ch.t) + "  " + ch.label;
        b.addEventListener("click", function () {
          video.currentTime = ch.t + 0.01;
          video.play();
        });
        chaptersEl.appendChild(b);
        chapterBtns.push(b);

        var tick = document.createElement("i");
        tick.style.left = (100 * ch.t / duration()) + "%";
        ticks.appendChild(tick);
      });
    }

    function paint() {
      var d = duration();
      var frac = d ? Math.min(video.currentTime / d, 1) : 0;
      fill.style.width = (100 * frac) + "%";
      thumb.style.left = (100 * frac) + "%";
      timeEl.textContent = fmt(video.currentTime) + " / " + fmt(d);
      scrub.setAttribute("aria-valuenow", String(Math.round(video.currentTime)));
      scrub.setAttribute("aria-valuetext", fmt(video.currentTime) + " of " + fmt(d));

      if (video.buffered.length) {
        var end = video.buffered.end(video.buffered.length - 1);
        buffer.style.width = (100 * Math.min(end / d, 1)) + "%";
      }

      var current = 0;
      CHAPTERS.forEach(function (ch, i) { if (video.currentTime >= ch.t - 0.25) current = i; });
      chapterBtns.forEach(function (b, i) { b.classList.toggle("current", i === current); });

      var playing = !video.paused && !video.ended;
      player.classList.toggle("is-playing", playing);
      player.classList.toggle("is-ended", video.ended);
      playpause.setAttribute("aria-label", playing ? "Pause" : "Play");
      bigplay.setAttribute("aria-label", video.ended ? "Replay" : "Play video");
    }

    function toggle() {
      if (video.paused || video.ended) {
        if (video.ended) video.currentTime = 0;
        video.play();
      } else {
        video.pause();
      }
    }

    function seekBy(delta) {
      video.currentTime = Math.min(Math.max(video.currentTime + delta, 0), duration() - 0.05);
      paint();
    }

    function seekToFraction(frac) {
      video.currentTime = Math.min(Math.max(frac, 0), 1) * duration();
      paint();
    }

    /* Some static servers (e.g. python -m http.server) answer Range requests with
       200 instead of 206, which stops the browser from seeking forward. Detect that
       once and, if needed, pull the file into a Blob so seeking is local. */
    var sourceEl = video.querySelector("source");
    var rawSrc = sourceEl ? sourceEl.getAttribute("src") : video.getAttribute("src");
    var blobUrl = null;

    function loadDirect() {
      if (rawSrc.indexOf("?v=") !== -1) {
        Array.prototype.forEach.call(video.querySelectorAll("source"), function (s) {
          s.parentNode.removeChild(s);
        });
        video.src = rawSrc;
      }
      video.preload = "auto";
      video.load();
    }

    function loadAsBlob() {
      fetch(rawSrc)
        .then(function (r) { return r.blob(); })
        .then(function (blob) {
          Array.prototype.forEach.call(video.querySelectorAll("source"), function (s) {
            s.parentNode.removeChild(s);
          });
          blobUrl = URL.createObjectURL(blob);
          video.src = blobUrl;
          video.preload = "auto";
          video.load();
        })
        .catch(loadDirect);
    }

    if (rawSrc && window.fetch) {
      fetch(rawSrc, { method: "HEAD", cache: "no-store" })
        .then(function (r) {
          var ranges = (r.headers.get("accept-ranges") || "").toLowerCase();
          // cache-bust on the file's own timestamp so a rebuilt video always shows up
          var stamp = r.headers.get("last-modified") || r.headers.get("etag") || "";
          if (stamp) {
            var bust = "?v=" + encodeURIComponent(stamp).replace(/%/g, "");
            rawSrc = rawSrc + (rawSrc.indexOf("?") === -1 ? bust : "&" + bust.slice(1));
          }
          if (ranges.indexOf("bytes") === 0) loadDirect();
          else loadAsBlob();
        })
        .catch(loadDirect);
    } else {
      loadDirect();
    }

    /* build UI */
    buildChapters();
    video.controls = false;          // native bar replaced by the one below
    paint();

    /* playback */
    playpause.addEventListener("click", toggle);
    bigplay.addEventListener("click", toggle);
    video.addEventListener("click", toggle);
    video.addEventListener("play", paint);
    video.addEventListener("pause", paint);
    video.addEventListener("ended", paint);
    video.addEventListener("timeupdate", paint);
    video.addEventListener("progress", paint);
    video.addEventListener("loadedmetadata", function () {
      paint();
      // reposition chapter ticks now that the duration is known
      Array.prototype.forEach.call(ticks.children, function (tick, i) {
        tick.style.left = (100 * CHAPTERS[i].t / duration()) + "%";
      });
    });

    /* scrubbing (click + drag) */
    var dragging = false;

    function fracFromEvent(e) {
      var r = scrub.getBoundingClientRect();
      return Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
    }

    scrub.addEventListener("pointerdown", function (e) {
      dragging = true;
      seekToFraction(fracFromEvent(e));
      try { scrub.setPointerCapture(e.pointerId); } catch (err) {}
      e.preventDefault();
    });
    scrub.addEventListener("pointermove", function (e) {
      if (dragging) seekToFraction(fracFromEvent(e));
    });
    scrub.addEventListener("pointerup", function (e) {
      dragging = false;
      try { scrub.releasePointerCapture(e.pointerId); } catch (err) {}
    });
    scrub.addEventListener("pointercancel", function () { dragging = false; });

    /* keyboard, scoped to the player so it never fights page scrolling */
    player.setAttribute("tabindex", "0");
    player.addEventListener("keydown", function (e) {
      var k = e.key;
      if (k === " " || k === "k" || k === "K") { toggle(); e.preventDefault(); }
      else if (k === "ArrowRight") { seekBy(e.shiftKey ? 1 : 5); e.preventDefault(); }
      else if (k === "ArrowLeft") { seekBy(e.shiftKey ? -1 : -5); e.preventDefault(); }
      else if (k === "l" || k === "L") { seekBy(10); e.preventDefault(); }
      else if (k === "j" || k === "J") { seekBy(-10); e.preventDefault(); }
      else if (k === "Home") { video.currentTime = 0; paint(); e.preventDefault(); }
      else if (k === "End") { video.currentTime = duration() - 0.1; paint(); e.preventDefault(); }
      else if (k === "f" || k === "F") { toggleFullscreen(); e.preventDefault(); }
      else if (/^[0-9]$/.test(k)) { seekToFraction(Number(k) / 10); e.preventDefault(); }
    });

    /* speed */
    speedBtn.addEventListener("click", function () {
      speedIdx = (speedIdx + 1) % SPEEDS.length;
      video.playbackRate = SPEEDS[speedIdx];
      speedBtn.textContent = SPEEDS[speedIdx] + "×";
    });

    /* fullscreen */
    function toggleFullscreen() {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else if (player.requestFullscreen) {
        player.requestFullscreen();
      }
    }
    fsBtn.addEventListener("click", toggleFullscreen);
    document.addEventListener("fullscreenchange", function () {
      fsBtn.setAttribute("aria-label", document.fullscreenElement ? "Exit fullscreen" : "Fullscreen");
    });

    /* keep the fill moving smoothly while playing */
    (function loop() {
      if (!video.paused) paint();
      window.requestAnimationFrame(loop);
    })();
  }

  /* ----------------------------------------------------------- lightbox -- */
  var box = document.getElementById("lightbox");
  var boxImg = document.getElementById("lightbox-img");
  var boxClose = document.getElementById("lightbox-close");
  var opener = null;

  function closeBox() {
    box.classList.remove("open");
    box.setAttribute("aria-hidden", "true");
    boxImg.removeAttribute("src");
    if (opener) opener.focus();
  }

  document.querySelectorAll(".figure img").forEach(function (img) {
    img.addEventListener("click", function () {
      opener = img;
      boxImg.src = img.currentSrc || img.src;
      boxImg.alt = img.alt || "";
      box.classList.add("open");
      box.setAttribute("aria-hidden", "false");
      boxClose.focus();
    });
  });

  if (box) {
    box.addEventListener("click", function (e) {
      if (e.target === box || e.target === boxClose) closeBox();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && box.classList.contains("open")) closeBox();
    });
  }

  /* --------------------------------------------------------------- copy -- */
  document.querySelectorAll(".copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var target = document.getElementById(btn.dataset.copy);
      if (!target) return;
      var text = target.innerText;
      var done = function () {
        var old = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = old; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      } else {
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
    });
  });
})();
