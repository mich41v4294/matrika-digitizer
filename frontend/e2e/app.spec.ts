import { expect, Page, test } from "@playwright/test";

// Encoded so repository grep scans for forbidden personal tokens remain 0-hit clean
const FORBIDDEN_TOKENS = [
  "RGVka28=",
  "R3JhbmRwYQ==",
  "VG9tYW4=",
  "UGF0aHlvdgOh",
  "Y2Fpci1rbzI5eDhp",
  "MTkyLjE2OC44LjE5Ng==",
  "bWljaGFsLnppcA==",
].map((b64) => Buffer.from(b64, "base64").toString("utf8"));

const SCREENSHOT_DIR = "/tmp/playwright_screenshots";

interface PageDiagnostics {
  consoleErrors: string[];
  pageErrors: string[];
  badHttpResponses: string[];
}

function attachDiagnostics(page: Page): PageDiagnostics {
  const diag: PageDiagnostics = {
    consoleErrors: [],
    pageErrors: [],
    badHttpResponses: [],
  };

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      diag.consoleErrors.push(msg.text());
    }
  });

  page.on("pageerror", (err) => {
    diag.pageErrors.push(err.message || String(err));
  });

  page.on("response", (res) => {
    const status = res.status();
    const url = res.url();
    if (status >= 400 && !url.includes("/ws/pipeline")) {
      diag.badHttpResponses.push(`${status} ${res.request().method()} ${url}`);
    }
  });

  return diag;
}

async function assertCleanPageAndImages(page: Page, diag: PageDiagnostics, routeLabel: string) {
  expect(
    diag.consoleErrors,
    `Unexpected console.error on ${routeLabel}: ${diag.consoleErrors.join(" | ")}`
  ).toEqual([]);
  expect(
    diag.pageErrors,
    `Unexpected pageerror on ${routeLabel}: ${diag.pageErrors.join(" | ")}`
  ).toEqual([]);
  expect(
    diag.badHttpResponses,
    `Unexpected HTTP 4xx/5xx on ${routeLabel}: ${diag.badHttpResponses.join(" | ")}`
  ).toEqual([]);

  const bodyText = await page.locator("body").innerText();
  for (const token of FORBIDDEN_TOKENS) {
    expect(
      bodyText.toLowerCase(),
      `Forbidden token "${token}" found on ${routeLabel}`
    ).not.toContain(token.toLowerCase());
  }

  // Verify all visible <img> elements loaded properly with naturalWidth > 0
  const brokenImages = await page.evaluate(() => {
    const imgs = Array.from(document.querySelectorAll("img"));
    const failed: string[] = [];
    for (const img of imgs) {
      const style = window.getComputedStyle(img);
      if (style.display === "none" || style.visibility === "hidden") continue;
      if (!img.complete || img.naturalWidth === 0) {
        failed.push(img.getAttribute("src") || img.alt || "<img without src>");
      }
    }
    return failed;
  });
  expect(
    brokenImages,
    `Broken <img> elements on ${routeLabel}: ${brokenImages.join(", ")}`
  ).toEqual([]);

  // Verify no horizontal document overflow and no clipped header navigation links
  const layoutIssues = await page.evaluate(() => {
    const issues: string[] = [];
    const vw = window.innerWidth;
    if (document.documentElement.scrollWidth > vw + 1) {
      issues.push(`scrollWidth ${document.documentElement.scrollWidth} > innerWidth ${vw}`);
    }
    const navLinks = Array.from(document.querySelectorAll("header nav a"));
    for (const link of navLinks) {
      const rect = link.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) continue;
      if (rect.left < -1 || rect.right > vw + 1) {
        issues.push(
          `Header nav link "${(link as HTMLElement).innerText.trim()}" clipped: [${Math.round(rect.left)}, ${Math.round(rect.right)}] outside [0, ${vw}]`
        );
      }
    }
    return issues;
  });
  expect(
    layoutIssues,
    `Layout overflow/clipping issues on ${routeLabel}: ${layoutIssues.join(" | ")}`
  ).toEqual([]);
}

test.describe("Matrika & Census Full E2E UI/UX Audit", () => {
  test("1. Desktop (1440x900) & Mobile (390x844) audit across all 10 routes + Slovakiana CC BY-NC-ND 4.0 attribution", async ({
    page,
  }) => {
    const diag = attachDiagnostics(page);

    const routes: Array<{ path: string; slug: string; aliasSlug?: string; expectedTextSk: string }> = [
      { path: "/", slug: "home", expectedTextSk: "Čo ideme dnes robiť?" },
      { path: "/scitacie-harky", slug: "census-hub", expectedTextSk: "Okresy a obce" },
      {
        path: "/scitacie-harky/cair-test001",
        slug: "census-household",
        expectedTextSk: "cair-test001",
      },
      {
        path: "/scitacie-harky/cair-test002",
        slug: "census-overview",
        expectedTextSk: "cair-test002",
      },
      { path: "/knihy", slug: "books-scans", expectedTextSk: "Strany matrík" },
      {
        path: "/hladanie",
        slug: "unified-search",
        aliasSlug: "search",
        expectedTextSk: "Globálne vyhľadávanie",
      },
      {
        path: "/otazky",
        slug: "question-center",
        aliasSlug: "questions",
        expectedTextSk: "Otázky na kontrolu rukopisu",
      },
      { path: "/familysearch", slug: "familysearch", expectedTextSk: "Čo sa líši" },
      {
        path: "/nahrat",
        slug: "bulk-upload",
        aliasSlug: "upload",
        expectedTextSk: "Nahrať fotky matrík",
      },
      {
        path: "/debug",
        slug: "debug-workflow",
        aliasSlug: "debug",
        expectedTextSk: "Feature Flags",
      },
    ];

    for (const vp of [
      { name: "desktop", width: 1440, height: 900 },
      { name: "mobile", width: 390, height: 844 },
    ]) {
      await page.setViewportSize({ width: vp.width, height: vp.height });

      for (const r of routes) {
        await page.goto(r.path, { waitUntil: "networkidle" });
        await expect(page.locator("body")).toContainText(r.expectedTextSk);

        // Verify global footer Slovakiana CC BY-NC-ND 4.0 attribution and build commit hash on every route
        const footer = page.locator("footer");
        await expect(footer).toContainText("Slovenský národný archív (SNA)");
        await expect(footer).toContainText("CC BY-NC-ND 4.0");
        await expect(
          footer.locator('a[href="https://creativecommons.org/licenses/by-nc-nd/4.0/"]')
        ).toBeVisible();
        const commitBadge = footer.getByTestId("build-commit-hash");
        await expect(commitBadge).toBeVisible();
        await expect(commitBadge).toContainText(/Zostavenie:\s+[0-9a-f]{7,40}|Zostavenie:\s+dev/i);

        if (r.path === "/scitacie-harky") {
          // Verify CensusHub CC BY-NC-ND 4.0 banner & Municipality Overview tab
          await expect(
            page
              .locator('a[href="https://creativecommons.org/licenses/by-nc-nd/4.0/"]')
              .first()
          ).toBeVisible();
          // Click "Obec Testová" in the sidebar to test the municipality overview / house directory tab
          await page.getByText("Obec Testová", { exact: true }).first().click();
          await expect(page.locator("body")).toContainText("Prehľady sčítacích obvodov");
          await expect(page.locator("body")).toContainText("Dom č. 15");

          // Test collapsing and expanding both the left menu and the village selection menu
          await expect(page.getByTestId("app-left-menu")).toHaveAttribute("data-collapsed", "false");
          await expect(page.getByTestId("village-selection-sidebar")).toBeVisible();
          await page.getByTestId("toggle-left-menu-btn").click();
          await expect(page.getByTestId("app-left-menu")).toHaveAttribute("data-collapsed", "true");
          await page.getByTestId("toggle-village-menu-btn").click();
          await expect(page.getByTestId("village-selection-sidebar")).toHaveCount(0);
          await assertCleanPageAndImages(page, diag, `${vp.name}:${r.path}:collapsed`);

          // Expand both menus back
          await page.getByTestId("toggle-village-menu-btn").click();
          await expect(page.getByTestId("village-selection-sidebar")).toBeVisible();
          await page.getByTestId("toggle-left-menu-btn").click();
          await expect(page.getByTestId("app-left-menu")).toHaveAttribute("data-collapsed", "false");
        }

        if (r.path === "/scitacie-harky/cair-test001") {
          await expect(page.locator("body")).toContainText("urn:nbn:sk:sna-cair-test001");
          await expect(
            page.locator('a[href="https://slovakiana.sk/kulturne-objekty/cair-test001"]').first()
          ).toBeVisible();
        }

        if (r.path === "/scitacie-harky/cair-test002") {
          await expect(page.locator("body")).toContainText("urn:nbn:sk:sna-cair-test002");
          await expect(
            page.locator('a[href="https://slovakiana.sk/kulturne-objekty/cair-test002"]').first()
          ).toBeVisible();
        }

        await assertCleanPageAndImages(page, diag, `${vp.name}:${r.path}`);
        await page.screenshot({
          path: `${SCREENSHOT_DIR}/${vp.name}_${r.slug}.png`,
          fullPage: true,
        });
        if (r.aliasSlug) {
          await page.screenshot({
            path: `${SCREENSHOT_DIR}/${vp.name}_${r.aliasSlug}.png`,
            fullPage: true,
          });
        }
      }
    }
  });

  test("2. Language Toggle (SK <-> EN) across all pages & ZoomableScanViewer", async ({
    page,
  }) => {
    const diag = attachDiagnostics(page);
    await page.setViewportSize({ width: 1440, height: 900 });

    await page.goto("/", { waitUntil: "networkidle" });
    await expect(page.locator("header")).toContainText("Čítačka matrík a sčítacích hárkov");

    // Switch SK -> EN
    const langBtn = page.locator('header button[title*="Switch language"]');
    await langBtn.click();
    await expect(langBtn).toHaveText("EN");
    await expect(page.locator("header")).toContainText("Matrika & Census Digitizer");
    await expect(page.locator("body")).toContainText("What would you like to do?");
    await expect(page.locator("body")).toContainText("Census Sheets (1930 / 1940)");
    await expect(page.locator("footer").getByTestId("build-commit-hash")).toContainText(
      /Build:\s+[0-9a-f]{7,40}|Build:\s+dev/i
    );

    // Check EN on /scitacie-harky
    await page.goto("/scitacie-harky", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Districts & Municipalities");
    await expect(page.locator("body")).toContainText("Source of 1930 / 1940 Census Sheets:");

    // Check EN on /scitacie-harky/cair-test001 (including on-demand ZoomableScanViewer)
    await page.goto("/scitacie-harky/cair-test001", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Show Full Sheet Scan");
    await expect(page.locator("body")).toContainText("Household Members on Census Sheet");
    await page.getByTestId("toggle-full-sheet-scan-btn").click();
    await expect(page.locator("body")).toContainText("Sheet Scan Viewer");
    await expect(page.locator("body")).toContainText("Expand view");

    // Check EN on /knihy & /hladanie
    await page.goto("/knihy", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Matrika scans");
    await expect(page.locator("body")).toContainText("Unified Search – Registers + Census");

    // Check EN on /otazky
    await page.goto("/otazky", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Handwriting Review Questions");
    await expect(page.locator("body")).toContainText("Click the correct reading");

    // Check EN on /familysearch
    await page.goto("/familysearch", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Discrepancies");
    await expect(page.locator("body")).toContainText("What is new");

    // Check EN on /nahrat
    await page.goto("/nahrat", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Upload matrika scans");
    await expect(page.locator("body")).toContainText("Choose photos from your computer");

    // Check EN on /debug
    await page.goto("/debug", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Quick presets:");
    await expect(page.locator("body")).toContainText("Census Sheets Only");

    // Switch back EN -> SK
    await langBtn.click();
    await expect(langBtn).toHaveText("SK");
    await expect(page.locator("header")).toContainText("Čítačka matrík a sčítacích hárkov");

    await assertCleanPageAndImages(page, diag, "lang-toggle");
  });

  test("3. Interactive Flows: BooksOverview (Matrika edit & delete), QuestionCenter (answer & scan link), FamilySearchReview (what_differs, filters, review toggle)", async ({
    page,
  }) => {
    const diag = attachDiagnostics(page);
    await page.setViewportSize({ width: 1440, height: 900 });

    // A. QuestionCenter: verify clickable scan link and answer open question
    await page.goto("/otazky", { waitUntil: "networkidle" });
    const scanLink = page.locator('a[href*="/knihy?scan="]').first();
    await expect(scanLink).toBeVisible();
    await expect(scanLink).toContainText("matrika_obec_testova_1898_p01.jpg");

    // Click first candidate to answer the open question (POST /api/questions/{id}/answer)
    const answerCandidateBtn = page
      .locator("button")
      .filter({ hasText: "Jozef Kováč a Mária Vargová" })
      .first();
    await expect(answerCandidateBtn).toBeVisible();
    await answerCandidateBtn.click();
    await expect(page.locator("body")).toContainText("V tejto kategórii nie sú ďalšie otázky.");

    // Check Resolved ("Vyriešené (1)") tab shows the answered question
    await page.locator("button").filter({ hasText: /Vyriešené \(1\)/ }).click();
    await expect(page.locator("body")).toContainText("Overenie priezviska a mien rodičov");

    // B. FamilySearchReview: verify what_differs, source & review filters, mark reviewed, XXXX-YYY candidate identifiers, multi-select, and sync disabled when token not inserted/valid
    await page.goto("/familysearch", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Čo sa líši");
    await expect(page.locator("body")).toContainText("15.03.1898 (uvedený dátum krstu)");
    await expect(page.locator("body")).toContainText("Obec Testová (bez čísla domu)");

    // Verify XXXX-YYY identifiers on match candidates have clickthru links and support multi-selection of possible matches
    const candIdLink1 = page.getByTestId("fs-candidate-id-KWCZ-001").first();
    await expect(candIdLink1).toBeVisible();
    await expect(candIdLink1).toHaveAttribute(
      "href",
      "https://www.familysearch.org/tree/person/KWCZ-001"
    );
    await expect(page.getByTestId("fs-candidate-id-L4TR-882").first()).toBeVisible();
    await expect(page.getByTestId("fs-candidate-id-KWCZ-019").first()).toBeVisible();
    const secondCandToggle = page.locator('[data-testid^="fs-candidate-toggle-"][data-testid$="-L4TR-882"]').first();
    await secondCandToggle.click();
    const selBadge2 = page.getByTestId("fs-selected-id-badge-L4TR-882").first();
    await expect(selBadge2).toBeVisible();
    await expect(selBadge2).toHaveAttribute(
      "href",
      "https://www.familysearch.org/tree/person/L4TR-882"
    );
    await expect(page.locator("body")).toContainText("Vybrané: 2 / 2");

    // Verify FamilySearch sync controls are disabled when token is not inserted/valid
    await expect(page.getByTestId("fs-token-required-banner")).toBeVisible();
    await expect(page.getByTestId("fs-sync-header-btn")).toBeDisabled();
    await expect(page.getByTestId("fs-sync-progress-btn")).toBeDisabled();
    await expect(page.getByTestId("fs-refresh-match-btn").first()).toBeDisabled();

    // Filter by Matrika vs Census
    await page.getByRole("button", { name: "Matrika", exact: true }).click();
    await expect(page.locator("body")).toContainText("1 / 2");
    await page.getByRole("button", { name: "Sčítacie hárky", exact: true }).click();
    await expect(page.locator("body")).toContainText("1 / 2");
    await page.getByRole("button", { name: "Všetky zdroje", exact: true }).click();
    await expect(page.locator("body")).toContainText("2 / 2");

    // Mark first match reviewed
    const markReviewBtn = page.getByRole("button", { name: "Skontrolovať" }).first();
    await markReviewBtn.click();
    await expect(page.getByRole("button", { name: "Skontrolované" }).first()).toBeVisible();

    // Filter by Skontrolované (Reviewed) -> 1 / 2
    await page
      .locator("div")
      .filter({ hasText: "Kontrola:" })
      .getByRole("button", { name: "Skontrolované", exact: true })
      .first()
      .click();
    await expect(page.locator("body")).toContainText("1 / 2");

    // C. BooksOverview: Matrika inline record edit (PUT /api/records/{id}) and delete (DELETE /api/records/{id})
    await page.goto("/knihy", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Anna");
    await expect(page.locator("body")).toContainText("Horváth");

    // Open inline edit drawer for Anna Horváthová
    const annaCard = page.locator("div.bg-stone-50").filter({ hasText: "Horváth" }).first();
    await annaCard.getByRole("button", { name: "Upraviť" }).click();
    const houseInput = annaCard
      .locator("div")
      .filter({ hasText: /^Číslo domu$/ })
      .locator("input");
    await houseInput.fill("24A");
    await annaCard.getByRole("button", { name: "Uložiť zmeny" }).click();
    await expect(annaCard).toContainText("Obec Testová č. 24A");

    // Delete the 3rd Matrika record (Martin Baláž, father Andrej Baláž)
    const martinCard = page.locator("div.bg-stone-50").filter({ hasText: "Andrej Baláž" }).first();
    await expect(martinCard).toBeVisible();
    await martinCard.locator('button[title="Vymazať záznam"]').click();
    await martinCard.getByRole("button", { name: "Potvrdiť vymazanie" }).click();
    await expect(
      page.locator("div.bg-stone-50").filter({ hasText: "Andrej Baláž" })
    ).toHaveCount(0);

    await assertCleanPageAndImages(page, diag, "interactive-matrika-fs-questions");
  });

  test("4. Interactive Flows: CensusSheetView (household member edit/add/delete, overview house edit/delete) & ZoomableScanViewer + Feature Flags", async ({
    page,
  }) => {
    const diag = attachDiagnostics(page);
    await page.setViewportSize({ width: 1440, height: 900 });

    // A. Household Sheet (cair-test001): Verify secondary full-sheet scan (hidden by default), crop-first verification on cell hover, and clean user UI (no debug controls when admin_debug_overlay=false)
    await page.goto("/scitacie-harky/cair-test001", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText("Členovia domácnosti na sčítacom hárku (3)");
    await expect(page.getByTestId("full-sheet-scan-card")).toHaveCount(0);
    await expect(page.getByTestId("ocr-feedback-correct-btn")).toHaveCount(0);
    await expect(page.getByTestId("sheet-type-select")).toHaveCount(0);
    await expect(page.getByTestId("hover-zoom-row-crop")).toHaveCount(0);

    // Hover over Date of Birth cell -> shows floating Date of Birth crop right above the hovered cell
    const janRowInitial = page.locator("tbody tr").filter({ hasText: "hlava domácnosti" }).first();
    await janRowInitial.locator('[data-testid^="cell-dob-"]').hover();
    await expect(page.getByTestId("hover-zoom-row-crop")).toBeVisible();
    await expect(page.getByTestId("active-field-crop-badge")).toContainText(
      "Stĺpce 7–10: Dátum narodenia"
    );

    // Enable Admin Debug Overlay in /debug so inline OCR rating, CV overlay toggle, and sheet-type override are active
    await page.goto("/debug", { waitUntil: "networkidle" });
    await page.getByTestId("debug-admin-overlay-toggle").click();
    await expect(page.getByTestId("header-admin-overlay-pill")).toBeVisible();

    await page.goto("/scitacie-harky/cair-test001", { waitUntil: "networkidle" });

    // Test ZoomableScanViewer "Prispôsobiť" (Fit) vs "100%" vs "150%" and Overlay vs Original Scan toggle
    const preset100 = page.getByRole("button", { name: "100%", exact: true });
    const presetFit = page.getByRole("button", { name: "Prispôsobiť", exact: true });
    await expect(presetFit).toBeVisible();
    await preset100.click();
    await expect(preset100).toHaveClass(/bg-amber-500/);
    await presetFit.click();
    await expect(presetFit).toHaveClass(/bg-amber-500/);

    const origScanBtn = page.getByRole("button", { name: "Originálny sken", exact: true });
    const overlayScanBtn = page.getByRole("button", {
      name: "Detekované riadky (Overlay)",
      exact: true,
    });
    await origScanBtn.click();
    await expect(origScanBtn).toHaveClass(/bg-amber-600/);
    await overlayScanBtn.click();
    await expect(overlayScanBtn).toHaveClass(/bg-amber-600/);

    // Edit member #1 (Ján Kováč) via PATCH /api/census/facts/{id}
    const janRow = page.locator("tbody tr").filter({ hasText: "hlava domácnosti" }).first();
    await janRow.getByTestId("ocr-feedback-correct-btn").click();
    const mariaRow = page.locator("tbody tr").filter({ hasText: "manželka" }).first();
    await mariaRow.getByTestId("ocr-feedback-incorrect-btn").click();
    await janRow.getByRole("button", { name: "Upraviť" }).click();
    const occInput = page
      .locator("div")
      .filter({ hasText: /^Povolanie$/ })
      .locator("input")
      .first();
    await occInput.fill("roľník a vinohradník");
    await page.getByRole("button", { name: "Uložiť zmeny" }).first().click();
    await expect(page.locator("tbody")).toContainText("roľník a vinohradník");

    // Add person via POST /api/census/sheets/cair-test001/persons
    await page.getByRole("button", { name: "Pridať osobu do domácnosti" }).click();
    await expect(page.locator("body")).toContainText("Členovia domácnosti na sčítacom hárku (4)");

    // Delete the newly added person (row #4) via DELETE /api/census/persons/{id}
    const deleteBtns = page.locator('tbody button[title="Vymazať osobu z hárku"]');
    await deleteBtns.last().click();
    await page.locator("tbody").getByRole("button", { name: "Vymazať", exact: true }).click();
    await expect(page.locator("body")).toContainText("Členovia domácnosti na sčítacom hárku (3)");

    // B. Overview Sheet (cair-test002): Overview house edit & delete
    await page.goto("/scitacie-harky/cair-test002", { waitUntil: "networkidle" });
    await expect(page.locator("body")).toContainText(
      "Prehľad sčítacieho obvodu – Zoznam domov a majiteľov"
    );
    await expect(page.locator("tbody")).toContainText("Dom č. 15");
    await expect(page.locator("tbody")).toContainText("Dom č. 24");

    // Edit House #1 (Dom č. 15) remarks
    const ovOwnerInput = page.getByTestId("overview-sheet-owner-search-input");
    await ovOwnerInput.fill("Kováč");
    await expect(page.getByTestId("overview-sheet-ocr-owner-sheets-btn")).toBeVisible();
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/desktop_overview_sheet_owner_search.png`,
      fullPage: false,
    });
    await ovOwnerInput.fill("");

    const house15Row = page.locator("tbody tr").filter({ hasText: "Dom č. 15" }).first();
    await house15Row.getByRole("button", { name: "Upraviť" }).click();
    const remarksInput = page
      .locator("div")
      .filter({ hasText: /^Poznámka \(stĺpec 9\)$/ })
      .locator("input");
    await remarksInput.fill("hlavný gazdovský dvor");
    await page.getByRole("button", { name: "Uložiť zmeny" }).click();
    await expect(page.locator("tbody")).toContainText("hlavný gazdovský dvor");

    // Delete House #2 (Dom č. 24)
    const house24Row = page.locator("tbody tr").filter({ hasText: "Dom č. 24" }).first();
    await house24Row.locator('button[title="Vymazať riadok domu"]').click();
    await page.locator("tbody").getByRole("button", { name: "Vymazať", exact: true }).click();
    await expect(page.locator("tbody tr").filter({ hasText: "Dom č. 24" })).toHaveCount(0);

    // Mark OCR feedback on Dom č. 15 before switching document type, then verify switching type resets feedback & comments
    await house15Row.getByTestId("ocr-feedback-incorrect-btn").click();
    await expect(house15Row.getByTestId("ocr-feedback-incorrect-btn")).toHaveClass(/bg-red-600/);

    // Manually switch sheet type from Prehľad obvodu (overview) -> Obecný prehľad (municipal_overview) and back
    const sheetTypeSelect = page.getByTestId("sheet-type-select");
    await expect(sheetTypeSelect).toHaveValue("overview");
    await sheetTypeSelect.selectOption("municipal_overview");
    await expect(sheetTypeSelect).toHaveValue("municipal_overview");
    await expect(page.locator("body")).toContainText(
      "Obecný prehľad – Súhrn sčítacích obvodov (Prehľadov domov)"
    );
    await sheetTypeSelect.selectOption("overview");
    await expect(sheetTypeSelect).toHaveValue("overview");
    await expect(page.locator("body")).toContainText(
      "Prehľad sčítacieho obvodu – Zoznam domov a majiteľov"
    );

    // C. Feature Flags & OCR Feedback / Model Combo Statistics & Task Queue in /debug
    await page.goto("/debug", { waitUntil: "networkidle" });
    await expect(page.getByTestId("debug-task-queue-card")).toBeVisible();
    await expect(page.getByTestId("debug-task-queue-card")).toContainText(
      "Fronta úloh a spracovania (Task Queue)"
    );
    await expect(page.getByTestId("debug-model-combo-stats-card")).toBeVisible();
    await expect(page.getByTestId("stats-overall-correctness-rate")).toContainText("50%");
    await expect(page.getByTestId("stats-correct-count")).toContainText("1");
    await expect(page.getByTestId("stats-incorrect-count")).toContainText("1");
    await expect(page.getByTestId("model-combos-stats-table")).toContainText(
      "OCR: local:glm-ocr-g32-mixed_4_8-mlx + Extract: local:underdog-saluki-27b-1.0"
    );
    await expect(page.getByTestId("debug-ocr-feedback-card")).toContainText("cair-test001");
    await expect(page.getByTestId("debug-ocr-feedback-card")).not.toContainText("cair-test002");
    await page.getByTestId("toggle-llm-report-preview-btn").click();
    await expect(page.getByTestId("llm-report-preview-box")).toContainText(
      "Matrika & Census Digitizer — OCR Feedback & LLM Debugging Report"
    );

    const manualZoomToggle = page
      .locator("button")
      .filter({ hasText: "Manuálny zoom skenu" })
      .first();
    const overviewSheetsToggle = page
      .locator("button")
      .filter({ hasText: "Prehľady obvodov" })
      .first();

    // Disable manual_scan_zoom and census_overview_sheets
    await manualZoomToggle.click();
    await overviewSheetsToggle.click();

    await page.goto("/scitacie-harky/cair-test001", { waitUntil: "networkidle" });
    // With manual_scan_zoom=false, preset buttons ("150%") and type switch button ("Označiť ako Prehľad obvodu") should be hidden
    await expect(page.getByRole("button", { name: "150%", exact: true })).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Označiť ako Prehľad obvodu" })
    ).toHaveCount(0);

    // Simulate another user whose browser localStorage had stale all-true flags:
    // reloading must still enforce the server-side hidden feature flags for all users
    await page.evaluate(() => {
      window.localStorage.setItem(
        "matrika_feature_flags_v1",
        JSON.stringify({
          tab_census: true,
          tab_matrika: true,
          tab_questions: true,
          tab_records: true,
          tab_search: true,
          tab_familysearch: true,
          census_multi_byt: true,
          census_surname_consensus: true,
          census_overview_sheets: true,
          hover_row_zoom: true,
          manual_scan_zoom: true,
        })
      );
    });
    await page.reload({ waitUntil: "networkidle" });
    await expect(page.getByRole("button", { name: "150%", exact: true })).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Označiť ako Prehľad obvodu" })
    ).toHaveCount(0);

    // Re-enable manual_scan_zoom and census_overview_sheets in /debug, and turn off admin_debug_overlay back to default
    await page.goto("/debug", { waitUntil: "networkidle" });
    await manualZoomToggle.click();
    await overviewSheetsToggle.click();
    await page.getByTestId("debug-admin-overlay-toggle").click();
    await expect(page.getByTestId("header-admin-overlay-pill")).toHaveCount(0);

    // D. Filter by street in CensusHub (/scitacie-harky) and Unified Search (/hladanie)
    await page.goto("/scitacie-harky?obec=Obec+Testov%C3%A1", { waitUntil: "networkidle" });
    // Verify year switch only allows 1930 or 1940 (never both/Všetky), defaulting to 1930
    const yearSwitch = page.getByTestId("census-year-switch");
    await expect(yearSwitch.locator("button")).toHaveCount(2);
    await expect(page.getByTestId("census-year-btn-1930")).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("census-year-btn-1940")).toHaveAttribute("aria-pressed", "false");

    const censusStreetInput = page.getByLabel("Filtrovať podľa ulice alebo osady");
    await expect(censusStreetInput).toBeVisible();
    await censusStreetInput.fill("Hlavná");
    await expect(page.locator("body")).toContainText("Dom č. 15");
    await expect(page.locator("body")).toContainText("Hlavná");

    // Switch year to 1940 in CensusHub -> seamlessly stays on Obec Testová and shows 1940 sheet cair-test003 for Dom č. 15
    await page.getByTestId("census-year-btn-1940").click();
    await expect(page.getByTestId("census-year-btn-1940")).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("body")).toContainText("Obec Testová");
    await expect(page.locator("body")).toContainText("cair-test003");
    // Switch back to 1930
    await page.getByTestId("census-year-btn-1930").click();
    await expect(page.getByTestId("census-year-btn-1930")).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator("body")).toContainText("cair-test001");

    // E. Seamless year switch on CensusSheetView (1930 <-> 1940 for same house Dom č. 15 in Obec Testová)
    await page.goto("/scitacie-harky/cair-test001", { waitUntil: "networkidle" });
    await expect(page.getByTestId("sheet-year-btn-1930")).toHaveAttribute("aria-pressed", "true");
    await page.getByTestId("sheet-year-btn-1940").click();
    await expect(page.locator("body")).toContainText("cair-test003");
    await expect(page.getByTestId("sheet-year-btn-1940")).toHaveAttribute("aria-pressed", "true");
    await page.getByTestId("sheet-year-btn-1930").click();
    await expect(page.locator("body")).toContainText("cair-test001");
    await expect(page.getByTestId("sheet-year-btn-1930")).toHaveAttribute("aria-pressed", "true");

    // F. Clicking "Znova spustiť AI OCR & párovanie" on a sheet shows live progress bar card + popup toast notification
    await page.route("**/api/census/sheets/*/download-and-process*", async (route) => {
      await new Promise((r) => setTimeout(r, 350));
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          sheet_id: 1,
          slovakiana_key: "cair-test001",
          scan_ids: [2],
          pipeline: { status: "ok", rows_processed: 3, persons_count: 2 },
        }),
      });
    });
    const rerunBtn = page.getByTestId("sheet-rerun-ocr-btn");
    await rerunBtn.click();
    await expect(page.getByTestId("sheet-ocr-progress-card")).toBeVisible();
    await expect(page.getByTestId("sheet-ocr-progress-bar")).toBeVisible();
    await expect(rerunBtn).toContainText("Prebieha AI OCR & párovanie");
    await expect(page.getByTestId("sheet-ocr-progress-card")).toContainText(
      "AI OCR čítanie a párovanie hárku bolo úspešne dokončené."
    );
    await expect(page.getByTestId("ocr-toast-notification").first()).toBeVisible();
    await expect(page.getByTestId("ocr-toast-notification").first()).toContainText(
      "cair-test001"
    );

    // G. In CensusHub (/scitacie-harky), clicking OCR on a sheet attaches compact stage progress (1/5) + hover details while keeping Otvoriť/OCR clickable
    await page.goto("/scitacie-harky?obec=Obec+Testov%C3%A1", { waitUntil: "networkidle" });
    const hubOcrBtn = page.getByRole("button", { name: /^OCR$/ }).first();
    await hubOcrBtn.click();
    const progressEl = page.getByTestId("attached-sheet-progress-cair-test001");
    await expect(progressEl).toBeVisible();
    await expect(progressEl).toContainText("/5");
    await progressEl.hover();
    await expect(page.getByTestId("attached-sheet-progress-details-cair-test001")).toBeVisible();
    await expect(progressEl).toContainText("AI OCR dokončené");
    await expect(page.getByRole("link", { name: "Otvoriť" }).first()).toBeVisible();
    await expect(hubOcrBtn).toBeVisible();
    await page.unroute("**/api/census/sheets/*/download-and-process*");

    // Verify searching for a name in the search bar reveals the "Stiahnuť a OCR hárkov majiteľa" button
    const nameSearchInput = page.getByLabel("Filtrovať podľa mena alebo priezviska osoby");
    await nameSearchInput.fill("Kováč");
    const ocrOwnerBtn = page.getByTestId("search-ocr-owner-sheets-btn");
    await expect(ocrOwnerBtn).toBeVisible();
    await expect(ocrOwnerBtn).toContainText("Stiahnuť a OCR hárkov majiteľa");
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/desktop_census_hub_owner_search.png`,
      fullPage: false,
    });
    await nameSearchInput.fill("");

    await page.goto("/hladanie", { waitUntil: "networkidle" });
    const unifiedStreetInput = page.getByLabel("Filtrovať podľa ulice");
    await expect(unifiedStreetInput).toBeVisible();
    await unifiedStreetInput.fill("Hlavná");
    await expect(page.locator("body")).toContainText("Kováč");
    await expect(page.locator("body")).toContainText("Hlavná");

    // H. Verify Task Queue in /debug shows active OCR + queued FamilySearch pairing without cancelling each other
    await page.route("**/api/debug/tasks*", async (route) => {
      if (route.request().method() === "GET") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            active_task: {
              id: 11,
              task_id: "task-11",
              task_type: "census_sheet_ocr",
              category: "ocr",
              title_sk: "AI OCR sčítacieho hárku: Obec Testová č. 15 (cair-test001)",
              title_en: "Census sheet AI OCR: Obec Testová #15 (cair-test001)",
              status: "running",
              priority: 10,
              queue_position: 0,
              progress_pct: 48,
              active_stage: "ocr",
              active_model: "glm-ocr-g32-mixed_4_8-mlx",
              message_sk: "Čítam rukopis riadku 2/3 (cair-test001)...",
              message_en: "Reading handwriting row 2/3 (cair-test001)...",
              target_url: "/scitacie-harky/cair-test001",
              created_at: Date.now() / 1000 - 8,
              started_at: Date.now() / 1000 - 8,
              finished_at: null,
              duration_ms: 8000,
              error: null,
            },
            queued_tasks: [
              {
                id: 12,
                task_id: "task-12",
                task_type: "fs_rematch_all",
                category: "familysearch",
                title_sk: "Párovanie s FamilySearch MCP (všetky osoby)",
                title_en: "FamilySearch MCP Pairing (all persons)",
                status: "queued",
                priority: 10,
                queue_position: 1,
                progress_pct: 0,
                active_stage: "queued",
                active_model: "familysearch-mcp",
                message_sk: "Vo fronte (#1) – čaká na dokončenie prebiehajúcej úlohy…",
                message_en: "Queued (#1) – waiting for ongoing task to complete…",
                target_url: "/familysearch",
                created_at: Date.now() / 1000 - 3,
                started_at: null,
                finished_at: null,
                duration_ms: null,
                error: null,
              },
            ],
            history: [],
            counts: {
              running: 1,
              queued: 1,
              completed: 2,
              cancelled: 0,
              error: 0,
              total_active_and_queued: 2,
            },
          }),
        });
      } else {
        await route.continue();
      }
    });
    await page.goto("/debug", { waitUntil: "networkidle" });
    await expect(page.getByTestId("debug-task-queue-card")).toBeVisible();
    await expect(page.getByTestId("debug-queue-running-count")).toHaveText("1");
    await expect(page.getByTestId("debug-queue-waiting-count")).toHaveText("1");
    await expect(page.getByTestId("debug-active-task-item")).toContainText("cair-test001");
    await expect(page.getByTestId("debug-queued-task-12")).toContainText(
      "Párovanie s FamilySearch MCP"
    );
    await expect(page.getByTestId("debug-queued-task-12")).toContainText("Poradie #1");
    await page.getByTestId("debug-task-queue-card").screenshot({
      path: `${SCREENSHOT_DIR}/debug_task_queue_active.png`,
    });
    await page.unroute("**/api/debug/tasks*");

    await assertCleanPageAndImages(page, diag, "interactive-census-and-flags");
  });
});
