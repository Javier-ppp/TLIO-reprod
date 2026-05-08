# TLIO Project Documentation - Directory Overview (`docs/`)

This document describes the structure and contents of the `docs/` directory, which hosts the project's website and landing page.

## Core Website Files

- `index.md`: The primary landing page. It provides a high-level overview of TLIO, including the abstract, publication details, system architecture diagrams, and performance results.
- `_config.yml`: Configuration file for the Jekyll static site generator. It specifies the theme (`bulma-clean-theme`), plugins, and site-wide metadata.
- `Gemfile`: Lists the Ruby dependencies (gems) needed to build and serve the documentation site locally.
- `node_modules/`: Contains frontend packages, primarily the Bulma CSS framework used for styling.

---

## Directory Structure

### `_includes/`
Contains reusable HTML snippets (partials) that are included in different layouts:
- `header.html` & `footer.html`: Global site navigation and footer.
- `hero.html`: The main banner section seen at the top of the index page.
- `showcase.html`: Handles the display of project features and media.
- `toc.html`: Implements the table of contents logic.
- `callouts.html`: Styles for highlighted notes or warnings.
- `google-analytics.html`: Script for website traffic tracking.

### `_layouts/`
Defines the structure for different types of pages:
- `default.html`: The master layout that wraps all other content.
- `page.html`: Layout for standalone pages (used by `index.md`).
- `post.html`: Layout for blog posts.
- `product.html` & `product-category.html`: Specialized layouts for showcasing specific features or results.

### `_sass/`
Contains the SCSS source files for the site's design:
- `_main.scss`: Global styles and overrides.
- `_layout.scss`: Structural styling for the site's grid and containers.
- `_showcase.scss`: Specific styling for the showcase and image sections.
- `syntax.scss`: Code highlighting styles.

### `assets/`
Storage for all static files used on the website:
- **Images**:
    - `cover.png`: The main hero image.
    - `system.png`: Diagram of the EKF and network interaction.
    - `IntroductionTrajAfterReview.png`: Comparison of trajectories.
    - `system-perf.png`, `err-sigmas.png`, `filter_ablation.png`: Performance result plots.
- **CSS & JS**:
    - `css/app.scss`: The entry point for compiled styles.
    - `js/app.js`: Client-side interactivity.
- **Documents**:
    - `fig9-simplified_proof.pdf`: Detailed mathematical proof related to the publication.

### `docs/` (Subdirectory)
- `_config.yml`: A secondary configuration file, likely used for sub-page generation or as a legacy artifact.
