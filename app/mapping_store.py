from dataclasses import dataclass


@dataclass(frozen=True)
class FAQItem:
    question: str
    answer: str


@dataclass(frozen=True)
class MappingProduct:
    slug: str
    title: str
    seo_title: str
    meta_description: str
    source_standard: str
    target_standard: str
    version_label: str
    price_usd: str
    polar_product_id: str
    sample_file_path: str
    sample_download_name: str
    sample_format_label: str
    paid_format_label: str
    hero_copy: str
    included_fields: tuple[str, ...]
    use_cases: tuple[str, ...]
    faq_items: tuple[FAQItem, ...]
    disclaimer: str
    related_slugs: tuple[str, ...]
    keywords: tuple[str, ...]
    updated_at: str
    featured: bool


MAPPING_PRODUCTS: dict[str, MappingProduct] = {
    "unspsc-to-cpv-mapping": MappingProduct(
        slug="unspsc-to-cpv-mapping",
        title="UNSPSC to CPV Mapping File",
        seo_title="UNSPSC to CPV Mapping File for EU Procurement | Classifast",
        meta_description=(
            "Map UNSPSC product and service codes to EU CPV procurement codes. "
            "Download a free sample and buy the full crosswalk file."
        ),
        source_standard="UNSPSC",
        target_standard="CPV",
        version_label="UNSPSC UNv260801.1 to CPV 2008 (2013 ver.)",
        price_usd="79",
        polar_product_id="replace-with-polar-product-id-unspsc-to-cpv",
        sample_file_path="app/static/mapping_samples/unspsc-to-cpv-sample.csv",
        sample_download_name="classifast-unspsc-to-cpv-sample.csv",
        sample_format_label="CSV sample",
        paid_format_label="Full CSV crosswalk",
        hero_copy=(
            "Turn internal UNSPSC-tagged catalogs into CPV-ready procurement data "
            "without rebuilding your taxonomy mapping from scratch."
        ),
        included_fields=(
            "UNSPSC code",
            "UNSPSC title",
            "CPV code",
            "CPV title",
            "Match rationale",
        ),
        use_cases=(
            "Prepare supplier or catalog data for EU tender workflows.",
            "Cross-reference global procurement taxonomies during ERP migration.",
            "Create analyst-ready lookup tables for spend normalization projects.",
        ),
        faq_items=(
            FAQItem(
                question="Who uses a UNSPSC to CPV crosswalk?",
                answer=(
                    "Procurement teams, bid managers, and data engineers use it "
                    "when supplier data is tagged in UNSPSC but downstream EU "
                    "tender systems require CPV codes."
                ),
            ),
            FAQItem(
                question="Does the file replace compliance review?",
                answer=(
                    "No. The file accelerates mapping work, but final public "
                    "procurement and compliance decisions still need human review."
                ),
            ),
        ),
        disclaimer=(
            "Reference mapping for operational acceleration. Validate critical "
            "matches before filing tenders, audits, or regulated submissions."
        ),
        related_slugs=("cpv-to-unspsc-mapping",),
        keywords=(
            "unspsc to cpv mapping",
            "unspsc cpv crosswalk",
            "cpv procurement mapping",
            "unspsc procurement crosswalk",
        ),
        updated_at="2026-03-16",
        featured=True,
    ),
    "cpv-to-unspsc-mapping": MappingProduct(
        slug="cpv-to-unspsc-mapping",
        title="CPV to UNSPSC Mapping File",
        seo_title="CPV to UNSPSC Mapping File for Supplier Data Cleanup | Classifast",
        meta_description=(
            "Convert EU CPV codes into UNSPSC-aligned categories for supplier "
            "master data, spend analysis, and catalog normalization."
        ),
        source_standard="CPV",
        target_standard="UNSPSC",
        version_label="CPV 2008 (2013 ver.) to UNSPSC UNv260801.1",
        price_usd="79",
        polar_product_id="replace-with-polar-product-id-cpv-to-unspsc",
        sample_file_path="app/static/mapping_samples/cpv-to-unspsc-sample.csv",
        sample_download_name="classifast-cpv-to-unspsc-sample.csv",
        sample_format_label="CSV sample",
        paid_format_label="Full CSV crosswalk",
        hero_copy=(
            "Translate CPV-coded tender or supplier data into UNSPSC so your "
            "catalogs, analytics, and procurement systems speak the same language."
        ),
        included_fields=(
            "CPV code",
            "CPV title",
            "UNSPSC code",
            "UNSPSC title",
            "Match rationale",
        ),
        use_cases=(
            "Normalize EU public procurement data into a global spend taxonomy.",
            "Prepare CPV-tagged exports for systems that index inventory by UNSPSC.",
            "Accelerate category enrichment projects for vendor or item masters.",
        ),
        faq_items=(
            FAQItem(
                question="Why would I map CPV into UNSPSC?",
                answer=(
                    "UNSPSC is often the internal procurement taxonomy in ERPs, "
                    "supplier portals, and spend analytics tools, while CPV appears "
                    "in EU public procurement datasets."
                ),
            ),
            FAQItem(
                question="Is the mapping one-to-one?",
                answer=(
                    "Not always. Some CPV concepts are broader or narrower than "
                    "individual UNSPSC nodes, so downstream review rules are still "
                    "recommended for critical use cases."
                ),
            ),
        ),
        disclaimer=(
            "Use as a crosswalk accelerator, not as a substitute for domain review "
            "in regulated procurement or reporting workflows."
        ),
        related_slugs=("unspsc-to-cpv-mapping",),
        keywords=(
            "cpv to unspsc mapping",
            "cpv unspsc crosswalk",
            "cpv code conversion",
            "unspsc supplier mapping",
        ),
        updated_at="2026-03-16",
        featured=True,
    ),
}


def list_mapping_products() -> list[MappingProduct]:
    return sorted(
        MAPPING_PRODUCTS.values(),
        key=lambda product: (not product.featured, product.title.lower()),
    )


def get_mapping_product(slug: str) -> MappingProduct | None:
    return MAPPING_PRODUCTS.get(slug)
