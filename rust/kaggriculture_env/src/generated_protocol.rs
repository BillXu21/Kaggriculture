// Generated from the installed Kaggriculture Conda environment. Do not edit by hand.
#![allow(dead_code)]
pub const ENGINE_VERSION: &str = "0.1.0";
pub const SOURCE_SHA256: &str = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e";
pub const SCHEMA_SHA256: &str = "a82c89c1a2315b93f39775d8e025471a01b738647c9772658368ee6b1b6f4867";
pub const RULE_TABLES_SHA256: &str = "afe76e9d01bcc55d9dc3906af22a8e2edd6cfb4fe67ad904b91ac3357d262c6c";
pub const BOARD_SIZE: usize = 10;
pub const MAX_HANDS: usize = 16;
pub const MAX_SHOP_INSTANCES: usize = 8;
pub const MAX_MARKET_ORDERS: usize = 10;
pub const MAX_QUANTITY: i64 = 100;
pub const SEASON_STEPS: usize = 720;
pub const SEASON_DAYS: usize = 30;
pub const TURNS_PER_DAY: usize = 24;
pub const UNIT_OPERATIONS: [&str; 18] = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "DROP", "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG"];
pub const MARKET_OPERATIONS: [&str; 7] = ["PASS", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"];
pub const CROPS: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
pub const PRODUCTS: [&str; 9] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"];
pub const ANIMALS: [&str; 3] = ["GOOSE", "COW", "SHEEP"];
pub const SHOPS: [&str; 8] = ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"];
pub const SHOP_ENGINE_ORDER: [&str; 8] = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"];
pub const CROP_SEED_COSTS: [i32; 5] = [10, 20, 50, 100, 80];
pub const CROP_FIRST_YIELD_DAY: [i32; 5] = [2, 2, 8, 10, 10];
pub const CROP_MAX_YIELD_DAY: [i32; 5] = [4, 3, 8, 10, 12];
pub const CROP_MAX_YIELD: [i32; 5] = [6, 4, 4, 4, 6];
pub const CROP_INTERVAL: [i32; 5] = [0, 0, 1, 2, 0];
pub const CROP_ONGOING: [bool; 5] = [false, false, true, true, false];
pub const ANIMAL_COSTS: [i32; 3] = [300, 400, 500];
pub const ANIMAL_STRUCTURE: [i8; 3] = [0, 1, 1];
pub const ANIMAL_FIRST_YIELD_DAY: [i32; 3] = [4, 8, 6];
pub const ANIMAL_INTERVAL: [i32; 3] = [1, 2, 3];
pub const ANIMAL_MAX_HELD: [i32; 3] = [4, 6, 6];
pub const ANIMAL_PRODUCT: [usize; 3] = [5, 6, 7];
pub const SHOP_DEMANDS: [[i8; 4]; 8] = [
    [5, 0, -1, -1],
    [5, 0, 3, -1],
    [0, 1, 2, 3],
    [3, 6, 0, -1],
    [1, -1, -1, -1],
    [6, 2, 0, -1],
    [3, 6, -1, -1],
    [7, -1, -1, -1],
];
pub const TOWN_CENTER_DEMAND_SCHEDULE: [[i64; 2]; 1] = [[0, 1]];
pub const LAND_PRICES: [i32; 3] = [1000, 2000, 4000];
pub const MARKET_I0: [f64; 9] = [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0];
pub const PRICE_FLOOR: i32 = 1;
pub const MARKET_BASE: [f64; 9] = [25.0, 35.0, 60.0, 120.0, 250.0, 50.0, 160.0, 200.0, 100.0];
pub const MARKET_T: [f64; 9] = [400.0, 450.0, 200.0, 100.0, 300.0, 332.0, 122.0, 105.0, 200.0];
pub const MARKET_BELOW_TARGET: [f64; 9] = [0.8, 1.0, 0.4, 0.7, 0.2, 0.4, 0.6, 0.2, 0.4];
pub const MARKET_ABOVE_TARGET: [f64; 9] = [0.2, 0.7, 0.6, 1.6, 3.6, 0.2, 1.6, 3.2, 0.4];
pub const MARKET_BELOW_SHAPE: [u8; 9] = [2, 5, 5, 2, 3, 5, 2, 3, 0];
pub const MARKET_ABOVE_SHAPE: [u8; 9] = [3, 2, 2, 0, 1, 3, 0, 1, 0];
pub const OBS_SIZE: usize = 5630;
pub const OBS_MARKET_INVENTORY: usize = 5540;
pub const OBS_MARKET_PRICES: usize = 5549;
pub const OBS_SHOPS: usize = 5560;
pub const OBS_SHED: usize = 5319;
pub const OBS_SEEDS: usize = 5331;
pub const OBS_INVENTORY: usize = 5336;
pub const OBS_ANIMAL_INVENTORY: usize = 5345;
pub const OBS_HAND_INVENTORY: usize = 5348;
pub const OBS_HAND_POSITIONS: usize = 5280;
pub const OBS_TILE_WIDTH: usize = 26;
pub const OBS_FARM_WIDTH: usize = 2600;
pub const NORMALIZE_MONEY: f32 = 10000.0;
pub const NORMALIZE_PRICE: f32 = 1000.0;
pub const NORMALIZE_COORDINATES: f32 = 9.0;
