from drift_happens.dataset.const import DATASET_DIR

AR23_TARGET_DIR = DATASET_DIR / "amazon_reviews_23"
AR23_CACHE_DIR = AR23_TARGET_DIR / "cache"
AR23_PREPROCESSED_DIR = AR23_TARGET_DIR / "processed"
AR23_PREPROCESSED_REVIEWS_DIR = AR23_PREPROCESSED_DIR / "reviews"
AR23_PREPROCESSED_REVIEWS_CACHE_FILE = AR23_PREPROCESSED_DIR / "reviews_cache.tar.gz"
AR23_PREPROCESSED_REVIEWS_MERGED_PATH = AR23_PREPROCESSED_DIR / "_merged.parquet"


AR23_GROUPS = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown",
]

# first 7 alphabetical categories, used for the conference pipeline experiments
AR23_GROUPS_SELECTED = AR23_GROUPS[:7]
