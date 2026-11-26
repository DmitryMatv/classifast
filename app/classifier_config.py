import os
from typing import NotRequired, TypedDict

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL",
    "qwen/qwen3-embedding-8b",
)
DEFAULT_EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "2048"))

# Query instructions follow the Qwen3-Embedding task pattern
# ("Given a web search query, retrieve relevant passages that answer the query"):
# name the kind of query and the kind of entry to retrieve. Documents are
# embedded without an instruction. Ranking rules belong in the reranker
# instructions below, since an embedding cannot apply them.

ETIM_QUERY_INSTRUCTION = (
    "Given a B2B technical product listing, retrieve the ETIM class that describes the type of product. "
    "The product's function and technical features identify the class."
)

UNSPSC_QUERY_INSTRUCTION = (
    "Given a description of a purchased product or service, retrieve the UNSPSC commodity that names it. "
    "Commodities are the most detailed UNSPSC level."
)

GPC_QUERY_INSTRUCTION = (
    "Given a trade item description, retrieve the GS1 GPC brick that categorizes the product. "
    "What the item essentially is and its intended use identify the brick."
)

EMDN_QUERY_INSTRUCTION = (
    "Given a medical device description, retrieve the terminal EMDN term that names its device type and intended purpose. "
    "Use hierarchy context, design, materials, and clinical application to separate similar terms."
)

NAICS_QUERY_INSTRUCTION = (
    "Given a description of a business or its activity, retrieve the NAICS industry that covers that activity. "
    "Manufacturing, wholesale, retail, and services are separate industries."
)

ISIC_QUERY_INSTRUCTION = (
    "Given a description of a business or economic activity, retrieve the ISIC class that covers that activity. "
    "Growing or extracting, manufacturing, trade, and services are separate classes."
)

NACE_QUERY_INSTRUCTION = (
    "Given a description of a business or economic activity, retrieve the NACE class that covers that activity. "
    "Growing or extracting, manufacturing, trade, and services are separate classes."
)

HS_QUERY_INSTRUCTION = (
    "Given a description of traded goods, retrieve the HS subheading that classifies them for customs. "
    "Material, processing state, and function identify the subheading."
)

CN_QUERY_INSTRUCTION = (
    "Given a description of traded goods, retrieve the EU Combined Nomenclature code that classifies them for customs. "
    "Material, processing state, and function identify the code."
)

HTS_QUERY_INSTRUCTION = (
    "Given a description of goods imported into the United States, retrieve the HTS tariff line that classifies them. "
    "Material, processing state, and function identify the line."
)

CPV_QUERY_INSTRUCTION = "Given a description of what a public buyer purchases, retrieve the CPV code that names those supplies, works, or services."

NSN_QUERY_INSTRUCTION = "Given a supply item description, retrieve NATO Stock Number items of the same type and specification."

# Reranker instructions follow the instruction-following reranker pattern
# (Voyage rerank-2.5+): state what the documents are, who writes the queries,
# and which entry should win. Rerankers without instruction support (such as
# Cohere Rerank 5) read this text as part of the query.

ETIM_RERANK_INSTRUCTION = (
    "Documents are ETIM classes for technical products such as electrical, lighting, HVAC, plumbing, and building equipment. "
    "Queries are B2B product listings, often with part numbers, brands, and ratings. "
    "Rank highest the class for the product type itself, judged by its function and technical features, not its brand or model number."
)

UNSPSC_RERANK_INSTRUCTION = (
    "Documents are UNSPSC entries, mostly commodities, used to classify purchases and spend. "
    "Queries describe a purchased product or service, often with brand or model names. "
    "Rank highest the commodity that names exactly this kind of product or service. "
    "Rank broad family or class entries lower, and also parts or accessories of the item unless the query itself is a part or accessory."
)

GPC_RERANK_INSTRUCTION = (
    "Documents are GS1 GPC bricks, the product categories used in retail and supplier catalogs. "
    "Queries are trade item descriptions, often with brand, size, or pack details. "
    "Rank highest the brick for what the item essentially is and its intended use, not its brand or packaging."
)

EMDN_RERANK_INSTRUCTION = (
    "Documents are terminal EMDN terms for medical devices, each shown with its ancestor hierarchy. "
    "Queries are device descriptions from manufacturers registering medical devices in the EU. "
    "Rank highest the term that matches the device type, intended purpose, and clinical application; "
    "use design and material details to choose between similar terms."
)

NAICS_RERANK_INSTRUCTION = (
    "Documents are NAICS 2022 entries at every level, from 2-digit sectors to 6-digit national industries. "
    "Queries describe a business or the activity it performs. "
    "Rank highest the most detailed industry that covers the main activity, and match its kind: "
    "manufacturing, wholesale, retail, and services are separate industries. Rank broader parent entries lower."
)

ISIC_RERANK_INSTRUCTION = (
    "Documents are ISIC classes, the most detailed level of the UN classification of economic activities, with their inclusion notes. "
    "Queries describe a business or the activity it performs. "
    "Rank highest the class for the main activity, and match its kind: "
    "growing or extracting, manufacturing, trade, and services are separate classes."
)

NACE_RERANK_INSTRUCTION = (
    "Documents are NACE Rev. 2.1 classes, the most detailed level of the EU classification of economic activities, with their inclusion notes. "
    "Queries describe a business or the activity it performs. "
    "Rank highest the class for the main activity, and match its kind: "
    "growing or extracting, manufacturing, trade, and services are separate classes."
)

HS_RERANK_INSTRUCTION = (
    "Documents are 6-digit Harmonized System subheadings used for customs classification. "
    "Queries describe traded goods. "
    "Rank highest the subheading that covers the goods as traded: their material, processing state "
    "(such as fresh, frozen, or prepared), and function. "
    "Rank parts, accessories, and related goods lower unless the query describes them."
)

CN_RERANK_INSTRUCTION = (
    "Documents are 8-digit EU Combined Nomenclature codes with their full customs descriptions. "
    "Queries describe goods in EU import, export, or intra-EU trade. "
    "Rank highest the code that covers the goods as traded: their material, processing state "
    "(such as fresh, frozen, or prepared), and function. "
    "Country of origin and brand names do not affect the code."
)

HTS_RERANK_INSTRUCTION = (
    "Documents are 10-digit US Harmonized Tariff Schedule lines, each shown as a path of descriptions "
    "from the heading down to the line; 'Other' covers goods not named by its sibling lines. "
    "Queries describe goods imported into the United States. "
    "Rank highest the line whose full path covers the goods as traded: their material, processing state, and function."
)

CPV_RERANK_INSTRUCTION = (
    "Documents are Common Procurement Vocabulary entries used in EU public tenders, at every level from divisions to detailed codes. "
    "Queries describe what a public buyer purchases or what a supplier provides. "
    "Rank highest the most specific entry that names these supplies, works, or services. Rank broader parent entries lower."
)

NSN_RERANK_INSTRUCTION = (
    "Documents are individual NATO Stock Number supply items, each with an item name and description. "
    "Queries describe a military or government supply item, sometimes with part details or ratings. "
    "Rank highest the item of the same type, then prefer matching specifications such as size, rating, and material."
)


class VersionConfig(TypedDict):
    collection_name: str
    base_url: NotRequired[str]
    append_code_to_url: NotRequired[bool]
    code_url_suffix: NotRequired[str]
    tooltip: NotRequired[str]


class ClassifierConfig(TypedDict):
    title: str
    heading: str
    description: str
    example: str
    embed_model_name: str
    embed_dims: int
    query_instruction: str
    rerank_instruction: NotRequired[str]
    query_subject: str
    versions: dict[str, VersionConfig]


CLASSIFIER_CONFIG: dict[str, ClassifierConfig] = {
    "ETIM": {
        "title": "ETIM International Classifier",
        "heading": "Find relevant EC classes from the ETIM standard.",
        "description": "ETIM (ETIM Technical Information Model) is a format to share and exchange product data based on taxonomic identification. This widely used classification standard for technical products was developed to structure the information flow between B2B professionals.",
        "example": "SH203-C20 Miniature Circuit Breaker 6kA 20A 3P",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": ETIM_QUERY_INSTRUCTION,
        "rerank_instruction": ETIM_RERANK_INSTRUCTION,
        "query_subject": "a technical product, such as electrical, lighting, HVAC, or plumbing equipment",
        "versions": {
            "ETIM version 10.0 (2024-12-10)": {
                "collection_name": "ETIM_10_1_Qwen3-8B_v1",
                "base_url": "https://viewer.etim-international.com/class/",
            },
        },
    },
    "UNSPSC": {
        "title": "UNSPSC Code Finder",
        "heading": "Get the right UNSPSC codes for your products and services.",
        "description": "The United Nations Standard Products and Services Code (UNSPSC) is a comprehensive, global classification system developed by the United Nations Development Programme (UNDP). This open, multi-sector standard enables organizations worldwide to classify products and services with precision and consistency. UNSPSC is essential for e-procurement platforms, supply chain optimization, spend analysis, vendor management, and facilitating B2B commerce across industries and borders.",
        "example": "Apple MacBook Air (13-inch, M3, 2024)",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": UNSPSC_QUERY_INSTRUCTION,
        "rerank_instruction": UNSPSC_RERANK_INSTRUCTION,
        "query_subject": "a purchased product or service",
        "versions": {
            "UNSPSC UNv260801.1 (18 March 2025)": {
                "collection_name": "UNSPSC_UNv260801-1__OR_Qwen3-8B__v2",
                "base_url": "https://usa.databasesets.com/unspsc/search?keywords=",
            },
        },
    },
    "GPC": {
        "title": "GS1 GPC Classifier",
        "heading": "Classify products by essential characteristics. Find the right GPC brick code.",
        "description": "GS1 Global Product Classification (GPC) is a product taxonomy used to group trade items by their essential characteristics. It organizes products from broad segments down to detailed bricks, helping retailers, suppliers, marketplaces, and data teams standardize product classification across catalog, procurement, analytics, and data synchronization workflows.",
        "example": "Oil spill absorbent pads",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": GPC_QUERY_INSTRUCTION,
        "rerank_instruction": GPC_RERANK_INSTRUCTION,
        "query_subject": "a retail or consumer trade item",
        "versions": {
            "GPC as of May 2026 (v20260520)": {
                "collection_name": "GPC_20260520_v1",
                "base_url": "https://gpc-browser.gs1.org/",
                "append_code_to_url": False,
            },
        },
    },
    "EMDN": {
        "title": "EMDN Code Finder",
        "heading": "Find the right EMDN code for a medical device.",
        "description": "The European Medical Device Nomenclature (EMDN) is used by manufacturers when registering medical devices in EUDAMED. This classifier uses the official English EMDN v2026 release and searches its 6 773 terminal terms that are the most granular assignable entries in the seven-level hierarchy.",
        "example": "Sterile single-use hypodermic syringe needle with an integrated safety mechanism",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": EMDN_QUERY_INSTRUCTION,
        "rerank_instruction": EMDN_RERANK_INSTRUCTION,
        "query_subject": "a medical device",
        "versions": {
            "EMDN v2026 (English)": {
                "collection_name": "EMDN_2026_EN_Qwen3-8B_v1",
                "base_url": "https://webgate.ec.europa.eu/dyna2/emdn/",
                "code_url_suffix": "#title",
            },
        },
    },
    "NAICS": {
        "title": "NAICS Code Finder",
        "heading": "Get the right NAICS codes for your business.",
        "description": "The North American Industry Classification System (NAICS) is the official industry classification system used by the United States, Canada, and Mexico to collect, analyze, and publish statistical data about their business economies. Developed jointly by these three countries, NAICS provides a standardized framework for measuring economic activity and is essential for business registration, tax reporting, government contracting, market research, and economic analysis across North America.",
        "example": "Dog grooming",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": NAICS_QUERY_INSTRUCTION,
        "rerank_instruction": NAICS_RERANK_INSTRUCTION,
        "query_subject": "a business or the economic activity it performs",
        "versions": {
            "2022 NAICS": {
                "collection_name": "NAICS_2022_eng_qwen3_8b_2048_v1",
                "base_url": "https://www.naics.com/code-search/?trms=",
                # "tooltip": "T = Canadian, Mexican, and United States industries are comparable",
            },
            # "2022 NAICS (only 6-digit codes)": {
            #    "collection_name": "NAICS_2022_SIXdigits_new001_v3",
            #    "base_url": "https://www.naics.com/code-search/?trms=",
            # },
        },
    },
    "ISIC": {
        "title": "ISIC Classifier",
        "heading": "Instantly categorize economic activities using the UN's ISIC.",
        "description": "The International Standard Industrial Classification of All Economic Activities (ISIC) is the global reference classification for economic activities developed by the United Nations Statistics Division. Used by national statistical offices worldwide, ISIC provides a comprehensive framework for organizing economic data by type of productive activity. It serves as the foundation for compiling national accounts, analyzing industrial statistics, and facilitating international comparisons of economic structure and performance across countries.",
        "example": "Gas station operation",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": ISIC_QUERY_INSTRUCTION,
        "rerank_instruction": ISIC_RERANK_INSTRUCTION,
        "query_subject": "a business or the economic activity it performs",
        "versions": {
            "ISIC Rev. 4": {
                "collection_name": "ISIC_Rev4_Qwen3-8B_v1",
                "base_url": "https://unstats.un.org/unsd/classifications/Econ/Structure/Detail/EN/27/",
            },
            "ISIC Rev. 5": {
                "collection_name": "ISIC_Rev5_Qwen3-8B_v1",
                "base_url": "https://unstats.un.org/unsd/classifications/Econ/Structure/Detail/EN/2029/",
            },
        },
    },
    "HS": {
        "title": "HS Code Finder",
        "heading": "Find the right HS (Harmonized System) codes for your goods.",
        "description": "The Harmonized Commodity Description and Coding System (HS) is a globally standardized nomenclature developed by the World Customs Organization (WCO) for classifying traded products. Used by over 200 countries and territories, the HS serves as the foundation for international trade statistics, customs tariffs, and trade negotiations. This six-digit classification system is essential for importers, exporters, customs brokers, and logistics professionals to determine applicable duties, taxes, trade restrictions, and regulatory requirements for goods crossing international borders.",
        "example": "Frozen mangoes",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": HS_QUERY_INSTRUCTION,
        "rerank_instruction": HS_RERANK_INSTRUCTION,
        "query_subject": "goods in international trade",
        "versions": {
            "HS6 2022": {
                "collection_name": "HS6_2022_Qwen3-8B_v1",
                "base_url": "https://www.tariffnumber.com/2026/",
            },
        },
    },
    "CN": {
        "title": "CN Code Finder",
        "heading": "Get CN (Combined Nomenclature) codes for EU customs and trade.",
        "description": "The Combined Nomenclature (CN) is the European Union's integrated tariff and statistical classification system, extending the international Harmonized System (HS) with EU-specific provisions. This 8-digit code structure is mandatory for all customs declarations, import/export documentation, and intra-EU trade statistics, serving as the legal basis for the EU's Common Customs Tariff and providing detailed classification for goods traded within the single market.",
        "example": "Frozen mangoes from Brazil",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": CN_QUERY_INSTRUCTION,
        "rerank_instruction": CN_RERANK_INSTRUCTION,
        "query_subject": "goods in EU import, export, or intra-EU trade",
        "versions": {
            "CN 2026": {
                "collection_name": "CN2026_Qwen3-8B_v4",
                "base_url": "https://www.tariffnumber.com/2026/",
            },
        },
    },
    "NACE": {
        "title": "NACE Business Activity Classifier",
        "heading": "Classify economic activities with EU's NACE standard.",
        "description": "NACE (Nomenclature statistique des activités économiques) is the European Union's statistical classification of economic activities, developed by Eurostat to ensure harmonized economic analysis across all EU member states. This comprehensive framework enables consistent business registration, national accounts compilation, employment statistics, and cross-country economic comparisons, serving as the foundation for EU policy-making, regional development planning, and structural business statistics.",
        "example": "Used car dealership",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": NACE_QUERY_INSTRUCTION,
        "rerank_instruction": NACE_RERANK_INSTRUCTION,
        "query_subject": "a business or the economic activity it performs",
        "versions": {
            "NACE Rev. 2.1": {
                "collection_name": "NACE_Rev2_1_Qwen3-8B_v2",
                "base_url": "https://showvoc.op.europa.eu/#/datasets/ESTAT_Statistical_Classification_of_Economic_Activities_in_the_European_Community_Rev._2.1._%28NACE_2.1%29/data?resId=http:%2F%2Fdata.europa.eu%2Fux2%2Fnace2.1%2F",
            },
        },
    },
    "CPV": {
        "title": "CPV Code Finder",
        "heading": "Find CPV codes for EU public procurement.",
        "description": "The Common Procurement Vocabulary (CPV) is the European Union's standardized classification system for public procurement, established to ensure transparency and equal access to public contracts across the single market. This 9-digit hierarchical code structure is mandatory for all EU public procurement procedures, enabling consistent tender documentation, contract award notices in the TED system, and comprehensive market analysis while facilitating cross-border bidding and ensuring compliance with EU procurement directives.",
        "example": "Indie gamedev studio",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": CPV_QUERY_INSTRUCTION,
        "rerank_instruction": CPV_RERANK_INSTRUCTION,
        "query_subject": "supplies, works, or services bought by a public authority",
        "versions": {
            "CPV 2008 (ver. 2013)": {
                "collection_name": "cpv_2008_qwen3_8b_v1",
            },
            "CPV 2008 Supplementary codes": {
                "collection_name": "cpv_2008_supplementary_qwen3_8b_v1",
            },
        },
    },
    "NSN": {
        "title": "NATO Stock Number (NSN) Classifier",
        "heading": "Find NSN (NATO Stock Number) codes for military procurement.",
        "description": "The NATO Stock Number (13 digits) consists of material group, material class, country code, and NIIN (National Item Identification Number). The NSN is a unique identifier for items of supply recognized by all NATO countries. The NSN is used to identify and manage supplies, ensuring that all member nations can effectively procure and utilize military equipment and materials. This classification system facilitates logistics, inventory management, and standardization across NATO forces.",
        "example": "1000W 120V AC power supply",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": NSN_QUERY_INSTRUCTION,
        "rerank_instruction": NSN_RERANK_INSTRUCTION,
        "query_subject": "a military or government supply item",
        "versions": {
            "NSN extract (February 22, 2023)": {
                "collection_name": "NSN_Qwen3-8B_v1",
            },
        },
    },
    "HTS": {
        "title": "HTS Code Finder",
        "heading": "Get Harmonized Tariff Schedule codes for US imports.",
        "description": "Harmonized Tariff Schedule (HTS) is the United States comprehensive customs classification system for imported goods, extending the international Harmonized System (HS) with country-specific provisions. This 10-digit hierarchical code structure is mandatory for all US customs declarations and serves as the legal basis for determining applicable duties, taxes, trade restrictions, and regulatory requirements. The HTS is essential for importers, customs brokers, freight forwarders, and compliance professionals to ensure accurate classification and smooth customs clearance for goods entering the United States.",
        "example": "Smartphone",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": HTS_QUERY_INSTRUCTION,
        "rerank_instruction": HTS_RERANK_INSTRUCTION,
        "query_subject": "goods imported into the United States",
        "versions": {
            "2026 HTS Revision 11 (July 1, 2026)": {
                "collection_name": "HTS_2026_Qwen3-8B_v1",
                "base_url": "https://hts.usitc.gov/search?query=",
            },
        },
    },
}
