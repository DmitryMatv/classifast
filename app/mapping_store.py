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
    image_url: str
    source_standard: str
    target_standard: str
    version_label: str
    price_usd: str
    polar_product_id: str
    sample_file_path: str
    sample_format_label: str
    paid_format_label: str
    hero_copy: str
    coverage_summary: str
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
        title="UNSPSC to CPV Mapping",
        seo_title="UNSPSC to CPV Mapping File | Classifast",
        meta_description=(
            "Direct UNSPSC to CPV mapping file covering UNSPSC Segments and Families (level 1+2, 617 categories) for procurement, finance, and reporting systems."
        ),
        image_url="https://classifast.com/static/images/preview.png",
        source_standard="UNSPSC",
        target_standard="CPV",
        version_label="UNSPSC UNv260801.1 -> CPV 2008",
        price_usd="265",
        polar_product_id="96ef9175-eb8b-408a-8e32-4fa803760f91",
        sample_file_path="app/static/mapping_samples/unspsc_to_cpv_mapping_sample.csv",
        sample_format_label="CSV sample",
        paid_format_label="Full CSV mapping file",
        hero_copy=(
            "Direct crosswalk from UNSPSC into CPV for organizations running different taxonomies across procurement systems, finance systems, ERP, and reporting workflows."
        ),
        coverage_summary=(
            "UNSPSC level 1+2 only, meaning only all Segments and all Families from UNSPSC UNv260801.1. Total scope is 617 categories. Class or Commodity levels are not used and are not mapped in this file."
        ),
        included_fields=(
            "UNSPSC code",
            "UNSPSC level (Segment or Family)",
            "UNSPSC title",
            "UNSPSC definition",
            "CPV code",
            "CPV title",
        ),
        use_cases=(
            "Align UNSPSC-coded finance or ERP masters with CPV-based procurement and tendering workflows.",
            "Build a level 1 and 2 reference crosswalk for spend analysis, reporting and category rollups.",
            "Avoid daisychaining UNSPSC to ProClass to CPV when a direct mapping file is the better fit.",
        ),
        faq_items=(
            FAQItem(
                question="Is this a full UNSPSC commodity-to-CPV mapping?",
                answer=(
                    "No. This file covers UNSPSC Segments and Families only (level 1+2). It maps 617 UNSPSC categories in total and does not attempt a full Class-level or Commodity-level crosswalk."
                ),
            ),
            FAQItem(
                question="Why use a direct UNSPSC to CPV mapping file?",
                answer=(
                    "A direct crosswalk reduces the category drift that can happen when teams convert through an intermediary taxonomy. It is useful when procurement and finance systems need a shared reference table across two coding standards."
                ),
            ),
        ),
        disclaimer=(
            "Reference mapping file for taxonomy harmonization across procurement, finance, and reporting systems. Validate material mappings before contract notices, regulated reporting, or audit-sensitive use."
        ),
        related_slugs=("cpv-to-unspsc-mapping",),
        keywords=(
            "unspsc to cpv mapping",
            "unspsc cpv crosswalk",
            "unspsc segments families cpv mapping",
            "procurement finance taxonomy crosswalk",
        ),
        updated_at="March 16, 2026",
        featured=True,
    ),
    "cpv-to-unspsc-mapping": MappingProduct(
        slug="cpv-to-unspsc-mapping",
        title="CPV to UNSPSC Mapping",
        seo_title="CPV to UNSPSC Mapping File | Classifast",
        meta_description=(
            "Convert CPV codes into UNSPSC-aligned categories for supplier master data, spend analysis, ERP harmonization, and catalog normalization."
        ),
        image_url="https://classifast.com/static/images/preview.png",
        source_standard="CPV",
        target_standard="UNSPSC",
        version_label="CPV 2008 -> UNSPSC UNv260801.1",
        price_usd="385",
        polar_product_id="cab2b78b-f1d6-4f2b-8859-dc95ca40a773",
        sample_file_path="app/static/mapping_samples/cpv_to_unspsc_mapping_sample.csv",
        sample_format_label="CSV sample",
        paid_format_label="Full CSV mapping file",
        hero_copy=(
            "Crosswalk CPV-coded tender, contract, and supplier data into UNSPSC so procurement, finance, ERP, and master-data systems can report against one internal taxonomy."
        ),
        coverage_summary=(
            "Each CPV 2008 code is mapped into the most relevant UNSPSC category for downstream normalization workflows. CPV code coverage is 100%."
        ),
        included_fields=(
            "CPV code",
            "UNSPSC code",
            "CPV title",
            "UNSPSC title",
            "CPV level (1-5)",
            "UNSPSC level",
        ),
        use_cases=(
            "Normalize TED or tender exports into UNSPSC for spend cube and category reporting.",
            "Align CPV coded supplier and contract data with UNSPSC-led ERP and P2P systems.",
            "Support taxonomy harmonization across procurement, finance and master data teams.",
        ),
        faq_items=(
            FAQItem(
                question="Why would I map CPV into UNSPSC?",
                answer=(
                    "UNSPSC is commonly the internal classification used in ERP, supplier master, P2P, and spend analytics environments, while CPV is the coding system attached to EU procurement notices."
                ),
            ),
            FAQItem(
                question="Is the mapping one-to-one?",
                answer=(
                    "Not always. Some CPV concepts are broader or narrower than the closest UNSPSC category, so review rules are still recommended for sensitive classification and reporting workflows."
                ),
            ),
        ),
        disclaimer=(
            "Use as a mapping accelerator, not as a substitute for domain review in regulated procurement, financial reporting, or compliance workflows."
        ),
        related_slugs=("unspsc-to-cpv-mapping",),
        keywords=(
            "cpv to unspsc mapping",
            "cpv unspsc crosswalk",
            "cpv code conversion",
            "unspsc supplier mapping",
        ),
        updated_at="March 16, 2026",
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
