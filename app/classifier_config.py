import os
from typing import NotRequired, TypedDict

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL",
    "Qwen/Qwen3-Embedding-8B",
)
DEFAULT_EMBEDDING_DIMS = int(os.getenv("HF_EMBEDDING_DIMS", "2048"))

# 'Given a web search query, retrieve relevant passages that answer the query'
DEFAULT_QUERY_INSTRUCTION = "Given a text description, retrieve the most relevant taxonomy entry that directly and specifically classifies it. Prefer exact functional, industry, material, or use-case matches over broad parent categories, and do not infer unsupported details."

UNSPSC_QUERY_INSTRUCTION = (
    "Retrieve the most relevant UNSPSC entry that directly and specifically classifies the given product or service description. "
    "Prefer commodity-level matches when available; avoid broad segment or family matches. Do not infer unsupported details."
)

DEFAULT_RERANK_INSTRUCTION = (
    "Determine if the candidate taxonomy entry directly and specifically classifies the text description. "
    "Score higher for exact functional, industry, material, or use-case matches. Penalize broad parent categories if a more specific match is possible. "
    "Do not reward inferences of unsupported details."
)

UNSPSC_RERANK_INSTRUCTION = (
    "Evaluate the match between the product/service description and the UNSPSC entry. "
    "Assign the highest scores to commodity-level matches. Strictly penalize broad segment or family matches unless no specific commodity match is supported. "
    "Do not infer unsupported details."
)


class VersionConfig(TypedDict):
    collection_name: str
    base_url: NotRequired[str]
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
    versions: dict[str, VersionConfig]


CLASSIFIER_CONFIG: dict[str, ClassifierConfig] = {
    "ETIM": {
        "title": "ETIM International Classifier",
        "heading": "Find relevant EC classes from the ETIM standard.",
        "description": "ETIM (ETIM Technical Information Model) is a format to share and exchange product data based on taxonomic identification. This widely used classification standard for technical products was developed to structure the information flow between B2B professionals.",
        "example": "SH203-C20 Miniature Circuit Breaker 6kA 20A 3P",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
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
        "versions": {
            "UNSPSC UNv260801.1 (18 March 2025)": {
                "collection_name": "UNSPSC_UNv260801-1__OR_Qwen3-8B__v2",
                "base_url": "https://usa.databasesets.com/unspsc/search?keywords=",
            },
        },
    },
    "GPC": {
        "title": "GPC Classifier",
        "heading": "Classify products by essential characteristics.",
        "description": "GS1 Global Product Classification (GPC) is a product taxonomy used to group trade items by their essential characteristics. It organizes products from broad segments down to detailed bricks, helping retailers, suppliers, marketplaces, and data teams standardize product classification across catalog, procurement, analytics, and data synchronization workflows.",
        "example": "Oil spill absorbent pads",
        "embed_model_name": DEFAULT_EMBEDDING_MODEL,
        "embed_dims": DEFAULT_EMBEDDING_DIMS,
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
        "versions": {
            "GPC as of May 2026 (v20260520)": {
                "collection_name": "GPC_20260520_v1",
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
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
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
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
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
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
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
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
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
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
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
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
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
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
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
        "query_instruction": DEFAULT_QUERY_INSTRUCTION,
        "rerank_instruction": DEFAULT_RERANK_INSTRUCTION,
        "versions": {
            "2026 HTS Revision 11 (July 1, 2026)": {
                "collection_name": "HTS_2026_Qwen3-8B_v1",
                "base_url": "https://hts.usitc.gov/search?query=",
            },
        },
    },
}
