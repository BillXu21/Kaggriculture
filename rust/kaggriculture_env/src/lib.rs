use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Arc;

use ndarray::{Array2, Array3};
use numpy::{
    PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray4, PyReadwriteArray2, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

#[rustfmt::skip]
mod generated_protocol;

const PLAYERS: usize = 2;
const MAX_HANDS: usize = generated_protocol::MAX_HANDS;
const BOARD_SIZE: usize = generated_protocol::BOARD_SIZE;
const TILE_COUNT: usize = BOARD_SIZE * BOARD_SIZE;
const PRODUCTS: usize = generated_protocol::PRODUCTS.len();
const CROP_TYPES: usize = generated_protocol::CROPS.len();
const ANIMAL_TYPES: usize = generated_protocol::ANIMALS.len();
const SHED_ITEMS: usize = PRODUCTS + ANIMAL_TYPES;
// Unit slots are farmer + MAX_HANDS hands; market slots follow them.
const MARKET_ACTION_START: usize = MAX_HANDS + 1;
const ACTION_SLOTS: usize = MARKET_ACTION_START + MAX_MARKET_ORDERS;
const ACTION_FIELDS: usize = 3;
const UNIT_MASK_WIDTH: usize = UNIT_OPERATIONS.len() + 17 + 101;
const MARKET_MASK_WIDTH: usize = MARKET_OPERATIONS.len() + 17 + 101;
const MASK_SIZE: usize = (MAX_HANDS + 1) * UNIT_MASK_WIDTH + MAX_MARKET_ORDERS * MARKET_MASK_WIDTH;
const MAX_MARKET_ORDERS: usize = generated_protocol::MAX_MARKET_ORDERS;
const MAX_QUANTITY: i64 = generated_protocol::MAX_QUANTITY;
const MAX_MARKET_LOOKUP_INVENTORY: usize = 20_000;
const ACTION_PLAYER_STRIDE: usize = ACTION_SLOTS * ACTION_FIELDS;
const ACTION_ENV_STRIDE: usize = PLAYERS * ACTION_PLAYER_STRIDE;
const OBSERVATION_SIZE: usize = generated_protocol::OBS_SIZE;
const OBS_MARKET_INVENTORY: usize = generated_protocol::OBS_MARKET_INVENTORY;
const OBS_MARKET_PRICES: usize = generated_protocol::OBS_MARKET_PRICES;
const OBS_SHOPS: usize = generated_protocol::OBS_SHOPS;
const OBS_SHED: usize = generated_protocol::OBS_SHED;
const OBS_SEEDS: usize = generated_protocol::OBS_SEEDS;
const OBS_INVENTORY: usize = generated_protocol::OBS_INVENTORY;
const OBS_ANIMAL_INVENTORY: usize = generated_protocol::OBS_ANIMAL_INVENTORY;
const OBS_HAND_INVENTORY: usize = generated_protocol::OBS_HAND_INVENTORY;
const OBS_HAND_POSITIONS: usize = generated_protocol::OBS_HAND_POSITIONS;
const OBS_FARM_BASE: usize = 62;
const SEASON_STEPS: f32 = generated_protocol::SEASON_STEPS as f32;
const SEASON_DAYS: f32 = generated_protocol::SEASON_DAYS as f32;
const TURNS_PER_DAY_NORMALIZED: f32 = generated_protocol::TURNS_PER_DAY as f32;
const NORMALIZE_MONEY: f32 = generated_protocol::NORMALIZE_MONEY;
const NORMALIZE_QUANTITY: f32 = generated_protocol::MAX_QUANTITY as f32;
const NORMALIZE_PRICE: f32 = generated_protocol::NORMALIZE_PRICE;
const NORMALIZE_COORDINATES: f32 = generated_protocol::NORMALIZE_COORDINATES;
const EMPTY_CROP_AGE: i32 = -1;
const EMPTY_KIND: i8 = -1;
const OBS_REMAINING_OVERAGE_NORMALIZED: f32 = 1.0;
const CROP_SEED_COSTS: [i32; CROP_TYPES] = generated_protocol::CROP_SEED_COSTS;
const ANIMAL_COSTS: [i32; ANIMAL_TYPES] = generated_protocol::ANIMAL_COSTS;
const ANIMAL_STRUCTURE: [i8; ANIMAL_TYPES] = generated_protocol::ANIMAL_STRUCTURE;
const ANIMAL_FIRST_YIELD_DAY: [i32; ANIMAL_TYPES] = generated_protocol::ANIMAL_FIRST_YIELD_DAY;
const ANIMAL_INTERVAL: [i32; ANIMAL_TYPES] = generated_protocol::ANIMAL_INTERVAL;
const ANIMAL_MAX_HELD: [i32; ANIMAL_TYPES] = generated_protocol::ANIMAL_MAX_HELD;
const ANIMAL_PRODUCT: [usize; ANIMAL_TYPES] = generated_protocol::ANIMAL_PRODUCT;
const SHOP_DEMANDS: [[i8; 4]; 8] = generated_protocol::SHOP_DEMANDS;
const TOWN_CENTER_DEMAND_SCHEDULE: [[i64; 2]; 1] = generated_protocol::TOWN_CENTER_DEMAND_SCHEDULE;
const MARKET_I0: [f64; PRODUCTS] = generated_protocol::MARKET_I0;
const MARKET_BASE: [f64; PRODUCTS] = generated_protocol::MARKET_BASE;
const MARKET_T: [f64; PRODUCTS] = generated_protocol::MARKET_T;
const MARKET_BELOW_TARGET: [f64; PRODUCTS] = generated_protocol::MARKET_BELOW_TARGET;
const MARKET_ABOVE_TARGET: [f64; PRODUCTS] = generated_protocol::MARKET_ABOVE_TARGET;
const MARKET_BELOW_SHAPE: [u8; PRODUCTS] = generated_protocol::MARKET_BELOW_SHAPE;
const MARKET_ABOVE_SHAPE: [u8; PRODUCTS] = generated_protocol::MARKET_ABOVE_SHAPE;
const CROP_FIRST_YIELD_DAY: [i32; CROP_TYPES] = generated_protocol::CROP_FIRST_YIELD_DAY;
const CROP_MAX_YIELD_DAY: [i32; CROP_TYPES] = generated_protocol::CROP_MAX_YIELD_DAY;
const CROP_MAX_YIELD: [i32; CROP_TYPES] = generated_protocol::CROP_MAX_YIELD;
const CROP_INTERVAL: [i32; CROP_TYPES] = generated_protocol::CROP_INTERVAL;
const LAND_PRICES: [i32; 3] = generated_protocol::LAND_PRICES;
const PRODUCT_NAMES: [&str; PRODUCTS] = generated_protocol::PRODUCTS;
const UNIT_OPERATIONS: [&str; 18] = generated_protocol::UNIT_OPERATIONS;
const MARKET_OPERATIONS: [&str; 7] = generated_protocol::MARKET_OPERATIONS;
const CROP_NAMES: [&str; CROP_TYPES] = generated_protocol::CROPS;
const ANIMAL_NAMES: [&str; ANIMAL_TYPES] = generated_protocol::ANIMALS;
const SHOPS: [&str; 8] = generated_protocol::SHOPS;
const SHOP_ENGINE_ORDER: [&str; 8] = generated_protocol::SHOP_ENGINE_ORDER;
const MAX_SHOP_INSTANCES: usize = generated_protocol::MAX_SHOP_INSTANCES;
const PRICE_FLOOR: i32 = generated_protocol::PRICE_FLOOR;
const PINNED_ENGINE_VERSION: &str = generated_protocol::ENGINE_VERSION;
const PINNED_SCHEMA_SHA256: &str = generated_protocol::SCHEMA_SHA256;
const PINNED_SOURCE_SHA256: &str = generated_protocol::SOURCE_SHA256;
const PINNED_RULE_TABLES_SHA256: &str = generated_protocol::RULE_TABLES_SHA256;

#[inline]
fn action_value(
    actions: &[i64],
    environment: usize,
    player: usize,
    slot: usize,
    field: usize,
) -> i64 {
    actions[environment * ACTION_ENV_STRIDE
        + player * ACTION_PLAYER_STRIDE
        + slot * ACTION_FIELDS
        + field]
}

#[inline]
fn tile_index(position: [i32; 2]) -> usize {
    (position[1] as usize) * BOARD_SIZE + position[0] as usize
}

#[inline]
fn move_position(position: &mut [i32; 2], operation: i64) {
    match operation {
        1 if position[1] > 0 => position[1] -= 1,
        2 if position[1] < BOARD_SIZE as i32 - 1 => position[1] += 1,
        3 if position[0] < BOARD_SIZE as i32 - 1 => position[0] += 1,
        4 if position[0] > 0 => position[0] -= 1,
        _ => {}
    }
}

type ObservationArray<'py> = Bound<'py, PyArray3<f32>>;
type RewardArray<'py> = Bound<'py, PyArray2<f32>>;
type StatusArray<'py> = Bound<'py, PyArray2<u8>>;
type ResetOutput<'py> = (ObservationArray<'py>, StatusArray<'py>);
type StepOutput<'py> = (ObservationArray<'py>, RewardArray<'py>, StatusArray<'py>);

#[derive(Clone)]
struct MarketConfig {
    base: [f64; PRODUCTS],
    i0: [f64; PRODUCTS],
    target: [f64; PRODUCTS],
    below_target: [f64; PRODUCTS],
    above_target: [f64; PRODUCTS],
    below_shape: [u8; PRODUCTS],
    above_shape: [u8; PRODUCTS],
}

impl Default for MarketConfig {
    fn default() -> Self {
        Self {
            base: MARKET_BASE,
            i0: MARKET_I0,
            target: MARKET_T,
            below_target: MARKET_BELOW_TARGET,
            above_target: MARKET_ABOVE_TARGET,
            below_shape: MARKET_BELOW_SHAPE,
            above_shape: MARKET_ABOVE_SHAPE,
        }
    }
}

impl MarketConfig {
    fn from_json(raw: &str) -> PyResult<Self> {
        let mut config = Self::default();
        let value: serde_json::Value = serde_json::from_str(raw)
            .map_err(|error| PyValueError::new_err(format!("invalid marketParams: {error}")))?;
        let Some(products) = value.as_object() else {
            return Err(PyValueError::new_err("marketParams must be an object"));
        };
        for (index, name) in PRODUCT_NAMES.iter().enumerate() {
            let Some(patch) = products.get(*name) else {
                continue;
            };
            let Some(patch) = patch.as_object() else {
                return Err(PyValueError::new_err(format!(
                    "marketParams.{name} must be an object"
                )));
            };
            if let Some(value) = patch.get("base") {
                config.base[index] = market_number(value, "base", name)?;
            }
            if let Some(value) = patch.get("I0") {
                config.i0[index] = market_number(value, "I0", name)?;
            }
            if let Some(value) = patch.get("T") {
                config.target[index] = market_number(value, "T", name)?;
            }
            if let Some(value) = patch.get("below_target") {
                config.below_target[index] = market_number(value, "below_target", name)?;
            }
            if let Some(value) = patch.get("above_target") {
                config.above_target[index] = market_number(value, "above_target", name)?;
            }
            if let Some(value) = patch.get("below_func") {
                config.below_shape[index] = market_shape_code(value, "below_func", name)?;
            }
            if let Some(value) = patch.get("above_func") {
                config.above_shape[index] = market_shape_code(value, "above_func", name)?;
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> PyResult<()> {
        for (index, name) in PRODUCT_NAMES.iter().enumerate() {
            if !self.base[index].is_finite() || self.base[index] < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "marketParams.{name}.base must be finite and non-negative"
                )));
            }
            if !self.i0[index].is_finite() || self.i0[index] <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "marketParams.{name}.I0 must be finite and positive"
                )));
            }
            if !self.target[index].is_finite() || self.target[index] <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "marketParams.{name}.T must be finite and positive"
                )));
            }
            for (field, value) in [
                ("below_target", self.below_target[index]),
                ("above_target", self.above_target[index]),
            ] {
                if !value.is_finite() || value < 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "marketParams.{name}.{field} must be finite and non-negative"
                    )));
                }
            }
        }
        Ok(())
    }
}

struct MarketLookupTables {
    prices: [Vec<i32>; PRODUCTS],
    buy_cost_prefix: [Vec<i64>; PRODUCTS],
    sell_revenue_prefix: [Vec<i64>; PRODUCTS],
    sell_supply_prefix: [Vec<i32>; PRODUCTS],
}

impl MarketLookupTables {
    fn build(config: &MarketConfig) -> Self {
        let prices = std::array::from_fn(|item| {
            (0..=MAX_MARKET_LOOKUP_INVENTORY)
                .map(|inventory| market_price_reference(config, item, inventory as f64))
                .collect::<Vec<_>>()
        });
        let buy_cost_prefix = std::array::from_fn(|item| {
            let mut prefix = vec![0_i64; MAX_MARKET_LOOKUP_INVENTORY + 1];
            for inventory in 0..MAX_MARKET_LOOKUP_INVENTORY {
                prefix[inventory + 1] = prefix[inventory] + prices[item][inventory] as i64;
            }
            prefix
        });
        let sell_revenue_prefix = std::array::from_fn(|item| {
            let mut prefix = vec![0_i64; MAX_MARKET_LOOKUP_INVENTORY + 1];
            for inventory in 0..MAX_MARKET_LOOKUP_INVENTORY {
                prefix[inventory + 1] = prefix[inventory] + prices[item][inventory] as i64;
            }
            prefix
        });
        let sell_supply_prefix = std::array::from_fn(|item| {
            let mut prefix = vec![0_i32; MAX_MARKET_LOOKUP_INVENTORY + 1];
            for inventory in 0..MAX_MARKET_LOOKUP_INVENTORY {
                prefix[inventory + 1] = prefix[inventory] + i32::from(prices[item][inventory] > 1);
            }
            prefix
        });
        Self {
            prices,
            buy_cost_prefix,
            sell_revenue_prefix,
            sell_supply_prefix,
        }
    }

    #[inline]
    fn get(&self, item: usize, inventory: f64) -> Option<i32> {
        if inventory.fract() != 0.0
            || !(0.0..=MAX_MARKET_LOOKUP_INVENTORY as f64).contains(&inventory)
        {
            return None;
        }
        Some(self.prices[item][inventory as usize])
    }

    #[inline]
    fn get_integer(&self, item: usize, inventory: i32) -> Option<i32> {
        let inventory = usize::try_from(inventory).ok()?;
        self.prices[item].get(inventory).copied()
    }

    #[inline]
    fn buy_cost_integer(&self, item: usize, inventory: i32, quantity: i32) -> Option<i64> {
        let inventory = usize::try_from(inventory).ok()?;
        let quantity = usize::try_from(quantity).ok()?;
        if quantity > inventory || inventory > MAX_MARKET_LOOKUP_INVENTORY {
            return None;
        }
        Some(
            self.buy_cost_prefix[item][inventory]
                - self.buy_cost_prefix[item][inventory - quantity],
        )
    }

    #[inline]
    fn sell_totals_integer(
        &self,
        item: usize,
        inventory: i32,
        quantity: i32,
    ) -> Option<(i64, i32)> {
        let inventory = usize::try_from(inventory).ok()?;
        let quantity = usize::try_from(quantity).ok()?;
        let end = inventory.checked_add(quantity)?;
        if end > MAX_MARKET_LOOKUP_INVENTORY {
            return None;
        }
        Some((
            self.sell_revenue_prefix[item][end] - self.sell_revenue_prefix[item][inventory],
            self.sell_supply_prefix[item][end] - self.sell_supply_prefix[item][inventory],
        ))
    }
}

#[derive(Clone)]
struct GameConfig {
    episode_steps: u32,
    turns_per_day: u32,
    weed_spawn_chance: f32,
    center_interval: u32,
    shop_sell_interval: u32,
    shop_unlock_interval_days: u32,
    starting_money: f32,
    max_market_orders: usize,
    shed_capacity: usize,
    farm_hand_cost_mult: i32,
    market_config: MarketConfig,
    market_tables: Arc<MarketLookupTables>,
}

fn market_number(value: &serde_json::Value, field: &str, product: &str) -> PyResult<f64> {
    value.as_f64().ok_or_else(|| {
        PyValueError::new_err(format!("marketParams.{product}.{field} must be a number"))
    })
}

fn market_shape_code(value: &serde_json::Value, field: &str, product: &str) -> PyResult<u8> {
    match value.as_str() {
        Some("linear") => Ok(0),
        Some("sq") => Ok(1),
        Some("sqrt") => Ok(2),
        Some("log") => Ok(3),
        Some("log10") => Ok(4),
        Some("hinge") => Ok(5),
        _ => Err(PyValueError::new_err(format!(
            "marketParams.{product}.{field} has an invalid shape"
        ))),
    }
}

fn config_u32(config: &serde_json::Map<String, serde_json::Value>, name: &str) -> PyResult<u32> {
    let value = config
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| PyValueError::new_err(format!("configuration.{name} must be an integer")))?;
    u32::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("configuration.{name} is too large")))
}

fn config_usize(
    config: &serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> PyResult<usize> {
    let value = config_u32(config, name)?;
    usize::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("configuration.{name} is too large")))
}

fn config_f32(config: &serde_json::Map<String, serde_json::Value>, name: &str) -> PyResult<f32> {
    let value = config
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| PyValueError::new_err(format!("configuration.{name} must be a number")))?;
    let value = value as f32;
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "configuration.{name} must be finite"
        )));
    }
    Ok(value)
}

fn validate_string_array(
    value: Option<&serde_json::Value>,
    name: &str,
    expected: &[&str],
) -> PyResult<()> {
    let actual = value
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| PyValueError::new_err(format!("rule_tables.{name} must be an array")))?;
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(value, expected)| value.as_str() != Some(*expected))
    {
        return Err(PyValueError::new_err(format!(
            "rule_tables.{name} does not match the compiled Rust vocabulary"
        )));
    }
    Ok(())
}

fn validate_i64_array(
    value: Option<&serde_json::Value>,
    name: &str,
    expected: &[i64],
) -> PyResult<()> {
    let actual = value
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| PyValueError::new_err(format!("rule_tables.{name} must be an array")))?;
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(value, expected)| value.as_i64() != Some(*expected))
    {
        return Err(PyValueError::new_err(format!(
            "rule_tables.{name} does not match the compiled Rust values"
        )));
    }
    Ok(())
}

fn rule_i64(
    object: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    field: &str,
) -> PyResult<i64> {
    object
        .get(field)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            PyValueError::new_err(format!("rule_tables.{name}.{field} must be an integer"))
        })
}

fn rule_string(
    object: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    field: &str,
) -> PyResult<String> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            PyValueError::new_err(format!("rule_tables.{name}.{field} must be a string"))
        })
}

fn validate_crop_tables(rules: &serde_json::Map<String, serde_json::Value>) -> PyResult<()> {
    let crops = rules
        .get("crops")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| PyValueError::new_err("rule_tables.crops must be an object"))?;
    for (index, name) in CROP_NAMES.iter().enumerate() {
        let crop = crops
            .get(*name)
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                PyValueError::new_err(format!("rule_tables.crops.{name} must be an object"))
            })?;
        let ongoing = generated_protocol::CROP_ONGOING[index];
        for (field, expected_value) in [
            ("seed", CROP_SEED_COSTS[index]),
            ("first_yield_day", CROP_FIRST_YIELD_DAY[index]),
            ("max_yield_day", CROP_MAX_YIELD_DAY[index]),
            ("interval", CROP_INTERVAL[index]),
            ("max_yield", CROP_MAX_YIELD[index]),
        ] {
            if rule_i64(crop, name, field)? != i64::from(expected_value) {
                return Err(PyValueError::new_err(format!(
                    "rule_tables.crops.{name}.{field} differs from Rust"
                )));
            }
        }
        if crop.get("ongoing").and_then(serde_json::Value::as_bool) != Some(ongoing) {
            return Err(PyValueError::new_err(format!(
                "rule_tables.crops.{name}.ongoing differs from Rust"
            )));
        }
    }
    Ok(())
}

fn validate_animal_tables(rules: &serde_json::Map<String, serde_json::Value>) -> PyResult<()> {
    let animals = rules
        .get("animals")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| PyValueError::new_err("rule_tables.animals must be an object"))?;
    for (index, name) in ANIMAL_NAMES.iter().enumerate() {
        let animal = animals
            .get(*name)
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                PyValueError::new_err(format!("rule_tables.animals.{name} must be an object"))
            })?;
        let structure = if ANIMAL_STRUCTURE[index] == 0 {
            "COOP"
        } else {
            "PASTURE"
        };
        let product = PRODUCT_NAMES[ANIMAL_PRODUCT[index]];
        for (field, expected_value) in [
            ("cost", ANIMAL_COSTS[index]),
            ("first_yield_day", ANIMAL_FIRST_YIELD_DAY[index]),
            ("interval", ANIMAL_INTERVAL[index]),
            ("max_held", ANIMAL_MAX_HELD[index]),
        ] {
            if rule_i64(animal, name, field)? != i64::from(expected_value) {
                return Err(PyValueError::new_err(format!(
                    "rule_tables.animals.{name}.{field} differs from Rust"
                )));
            }
        }
        if rule_string(animal, name, "structure")? != structure
            || rule_string(animal, name, "product")? != product
        {
            return Err(PyValueError::new_err(format!(
                "rule_tables.animals.{name} differs from Rust"
            )));
        }
    }
    Ok(())
}

fn validate_shop_tables(rules: &serde_json::Map<String, serde_json::Value>) -> PyResult<()> {
    let shops = rules
        .get("shops")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| PyValueError::new_err("rule_tables.shops must be an object"))?;
    for (shop_index, name) in SHOP_ENGINE_ORDER.iter().enumerate() {
        let products = shops
            .get(*name)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                PyValueError::new_err(format!("rule_tables.shops.{name} must be an array"))
            })?;
        let expected_products: Vec<&str> = SHOP_DEMANDS[shop_index]
            .iter()
            .copied()
            .filter(|index| *index >= 0)
            .map(|index| PRODUCT_NAMES[index as usize])
            .collect();
        if products.len() != expected_products.len()
            || products
                .iter()
                .zip(expected_products)
                .any(|(actual, expected)| actual.as_str() != Some(expected))
        {
            return Err(PyValueError::new_err(format!(
                "rule_tables.shops.{name} differs from Rust"
            )));
        }
    }
    Ok(())
}

fn validate_rule_tables(payload: &serde_json::Value) -> PyResult<()> {
    let rules = payload
        .get("rule_tables")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| PyValueError::new_err("resolved config must contain rule_tables"))?;
    validate_string_array(rules.get("products"), "products", &PRODUCT_NAMES)?;
    validate_string_array(
        rules.get("unit_operations"),
        "unit_operations",
        &UNIT_OPERATIONS,
    )?;
    validate_string_array(
        rules.get("market_operations"),
        "market_operations",
        &MARKET_OPERATIONS,
    )?;
    validate_crop_tables(rules)?;
    validate_animal_tables(rules)?;
    validate_shop_tables(rules)?;
    validate_string_array(rules.get("land_order"), "land_order", &["NE", "SW", "SE"])?;
    let land_prices: [i64; 3] = LAND_PRICES.map(i64::from);
    validate_i64_array(rules.get("land_prices"), "land_prices", &land_prices)?;
    validate_string_array(
        rules.get("town_center_products"),
        "town_center_products",
        &PRODUCT_NAMES[..8],
    )?;
    if rules
        .get("max_shop_instances")
        .and_then(serde_json::Value::as_u64)
        != Some(8)
    {
        return Err(PyValueError::new_err(
            "rule_tables.max_shop_instances does not match Rust",
        ));
    }
    let demand_schedule = rules
        .get("town_center_demand_schedule")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            PyValueError::new_err("rule_tables.town_center_demand_schedule must be an array")
        })?;
    if demand_schedule.len() != 1
        || demand_schedule[0].as_array().map(Vec::len) != Some(2)
        || demand_schedule[0][0].as_i64() != Some(TOWN_CENTER_DEMAND_SCHEDULE[0][0])
        || demand_schedule[0][1].as_i64() != Some(TOWN_CENTER_DEMAND_SCHEDULE[0][1])
    {
        return Err(PyValueError::new_err(
            "rule_tables.town_center_demand_schedule does not match Rust",
        ));
    }
    for (name, expected) in [
        ("crops", &CROP_NAMES[..]),
        ("animals", &ANIMAL_NAMES[..]),
        ("shops", &SHOPS[..]),
    ] {
        let object = rules
            .get(name)
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                PyValueError::new_err(format!("rule_tables.{name} must be an object"))
            })?;
        if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
            return Err(PyValueError::new_err(format!(
                "rule_tables.{name} does not match the compiled Rust vocabulary"
            )));
        }
    }
    for (name, expected) in [
        ("max_hands", MAX_HANDS as u64),
        ("max_market_orders", MAX_MARKET_ORDERS as u64),
        ("max_quantity", MAX_QUANTITY as u64),
        ("board_size", BOARD_SIZE as u64),
    ] {
        if rules.get(name).and_then(serde_json::Value::as_u64) != Some(expected) {
            return Err(PyValueError::new_err(format!(
                "rule_tables.{name} does not match the compiled Rust value"
            )));
        }
    }
    Ok(())
}

fn market_shape(code: u8, x: f64, t: f64) -> f64 {
    let x = x.max(0.0);
    match code {
        0 => x,
        1 => x * x,
        2 => x.sqrt(),
        3 => (1.0 + x).ln(),
        4 => (1.0 + x).log10(),
        5 if t > 0.0 => {
            let u = x / t;
            u + 8.0 * (u - 1.0).max(0.0).powi(2)
        }
        5 => x,
        _ => 0.0,
    }
}

fn python_round(value: f64) -> i32 {
    let lower = value.floor();
    let fraction = value - lower;
    let rounded = if fraction < 0.5 {
        lower
    } else if fraction > 0.5 {
        lower + 1.0
    } else if (lower as i64) % 2 == 0 {
        lower
    } else {
        lower + 1.0
    };
    rounded as i32
}

fn market_price_reference(config: &MarketConfig, item: usize, inventory: f64) -> i32 {
    let base = config.base[item];
    let i0 = config.i0[item];
    let target = if inventory < i0 {
        let shape = market_shape(
            config.below_shape[item],
            i0 - inventory,
            config.target[item],
        );
        let endpoint = market_shape(
            config.below_shape[item],
            config.target[item],
            config.target[item],
        );
        base + config.below_target[item] * base / endpoint * shape
    } else {
        let shape = market_shape(
            config.above_shape[item],
            inventory - i0,
            config.target[item],
        );
        let endpoint = market_shape(
            config.above_shape[item],
            config.target[item],
            config.target[item],
        );
        base - config.above_target[item] * base / endpoint * shape
    };
    python_round(target.max(PRICE_FLOOR as f64))
}

fn fibonacci(index: usize) -> i32 {
    let (mut first, mut second) = (1_i32, 1_i32);
    for _ in 0..index {
        (first, second) = (second, first + second);
    }
    first
}

struct PythonRandom {
    state: [u32; 624],
    index: usize,
}

impl PythonRandom {
    fn new(seed: u128) -> Self {
        let key = [seed as u32, (seed >> 32) as u32, (seed >> 64) as u32];
        let key_len = if key[2] != 0 {
            3
        } else if key[1] != 0 {
            2
        } else {
            1
        };
        let mut state = [0u32; 624];
        state[0] = 19_650_218;
        for index in 1..624 {
            state[index] = 1_812_433_253u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        let mut i = 1usize;
        let mut j = 0usize;
        let mut count = 624usize.max(key_len);
        while count > 0 {
            state[i] = (state[i] ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
            if j >= key_len {
                j = 0;
            }
            count -= 1;
        }
        count = 623;
        while count > 0 {
            state[i] = (state[i]
                ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                state[0] = state[623];
                i = 1;
            }
            count -= 1;
        }
        state[0] = 0x8000_0000;
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624] ^ (value >> 1);
                if value & 1 != 0 {
                    self.state[index] ^= 0x9908_b0df;
                }
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn random(&mut self) -> f32 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        ((a * 67_108_864.0 + b) / 9_007_199_254_740_992.0) as f32
    }

    fn choice(&mut self, length: usize) -> usize {
        let bits = usize::BITS - length.leading_zeros();
        loop {
            let value = (self.next_u32() >> (32 - bits)) as usize;
            if value < length {
                return value;
            }
        }
    }
}

#[derive(Clone)]
struct GameState {
    step: u32,
    done: bool,
    episode_steps: u32,
    turns_per_day: u32,
    weed_spawn_chance: f32,
    center_interval: u32,
    shop_sell_interval: u32,
    shop_unlock_interval_days: u32,
    max_market_orders: usize,
    shed_capacity: usize,
    farm_hand_cost_mult: i32,
    market_config: MarketConfig,
    episode_seed: u64,
    positions: [[i32; 2]; PLAYERS],
    hand_positions: [[[i32; 2]; MAX_HANDS]; PLAYERS],
    hand_count: [usize; PLAYERS],
    hires_today: [usize; PLAYERS],
    unlocked: [[bool; 4]; PLAYERS],
    money: [f32; PLAYERS],
    crop_age: [[i32; TILE_COUNT]; PLAYERS],
    crop_kind: [[i8; TILE_COUNT]; PLAYERS],
    crop_decay_step: [[i32; TILE_COUNT]; PLAYERS],
    crop_yield: [[i32; TILE_COUNT]; PLAYERS],
    crop_watered_today: [[bool; TILE_COUNT]; PLAYERS],
    crop_unwatered: [[i32; TILE_COUNT]; PLAYERS],
    crop_fertilized_until_day: [[i32; TILE_COUNT]; PLAYERS],
    seeds: [[i32; CROP_TYPES]; PLAYERS],
    animal_fed: [[bool; TILE_COUNT]; PLAYERS],
    animal_age: [[i32; TILE_COUNT]; PLAYERS],
    animal_yield: [[i32; TILE_COUNT]; PLAYERS],
    structure: [[bool; TILE_COUNT]; PLAYERS],
    structure_kind: [[i8; TILE_COUNT]; PLAYERS],
    animal_on_tile: [[bool; TILE_COUNT]; PLAYERS],
    animal_kind: [[i8; TILE_COUNT]; PLAYERS],
    animal_unfed: [[i32; TILE_COUNT]; PLAYERS],
    animal_cared: [[bool; TILE_COUNT]; PLAYERS],
    animal_fertilizer_available: [[bool; TILE_COUNT]; PLAYERS],
    animal_pending_care: [[i32; TILE_COUNT]; PLAYERS],
    weed: [[bool; TILE_COUNT]; PLAYERS],
    shop_instances: [i8; MAX_SHOP_INSTANCES],
    shop_count: usize,
    shed: [[i32; SHED_ITEMS]; PLAYERS],
    shed_used: [i32; PLAYERS],
    inventory: [[i32; 9]; PLAYERS],
    animal_inventory: [[i32; ANIMAL_TYPES]; PLAYERS],
    hand_inventory: [[[i32; 9]; MAX_HANDS]; PLAYERS],
    hand_animal_inventory: [[[i32; 3]; MAX_HANDS]; PLAYERS],
    // Insertion-order stamps for each entity's combined 12-slot carried
    // inventory (products 0-8, animals 9-11): 0 means the item isn't
    // currently carried, otherwise the value of `item_order_seq` at the
    // moment it was last added from empty. Python's DROP iterates the
    // farmer's inventory `dict` in insertion order and recomputes shed
    // room each step, so whichever item entered the dict first gets
    // priority for scarce shed room; a fixed products-then-animals
    // iteration order (as DROP used to use) diverges whenever a unit
    // carries 2+ item types and the shed can't fit all of them. See
    // finding 11 in docs/rl/RUST_ENGINE_PARITY_FINDINGS.md.
    item_order_seq: u32,
    farmer_item_order: [[u32; SHED_ITEMS]; PLAYERS],
    hand_item_order: [[[u32; SHED_ITEMS]; MAX_HANDS]; PLAYERS],
    market_inventory: [f64; PRODUCTS],
    market_prices: [i32; PRODUCTS],
    market_dirty: u16,
    market_tables: Arc<MarketLookupTables>,
}

#[derive(Clone, Copy)]
enum EntityInventoryRef {
    Farmer,
    Hand(usize),
}

#[derive(Clone, Copy)]
struct ActiveUnit {
    position: [i32; 2],
    inventory: [i32; PRODUCTS],
    animal_inventory: [i32; ANIMAL_TYPES],
    item_order: [u32; SHED_ITEMS],
}

impl GameState {
    fn with_config(seed: u64, config: GameConfig) -> Self {
        let GameConfig {
            episode_steps,
            turns_per_day,
            weed_spawn_chance,
            center_interval,
            shop_sell_interval,
            shop_unlock_interval_days,
            starting_money,
            max_market_orders,
            shed_capacity,
            farm_hand_cost_mult,
            market_config,
            market_tables,
        } = config;
        let mut state = Self {
            step: 0,
            done: false,
            episode_steps,
            turns_per_day,
            weed_spawn_chance,
            center_interval,
            shop_sell_interval,
            shop_unlock_interval_days,
            max_market_orders,
            shed_capacity,
            farm_hand_cost_mult,
            episode_seed: seed,
            positions: [[4, 4], [4, 4]],
            hand_positions: [[[0, 0]; MAX_HANDS]; PLAYERS],
            hand_count: [0; PLAYERS],
            hires_today: [0; PLAYERS],
            unlocked: [[true, false, false, false]; PLAYERS],
            money: [starting_money; PLAYERS],
            crop_age: [[EMPTY_CROP_AGE; TILE_COUNT]; PLAYERS],
            crop_kind: [[EMPTY_KIND; TILE_COUNT]; PLAYERS],
            crop_decay_step: [[-1; TILE_COUNT]; PLAYERS],
            crop_yield: [[0; TILE_COUNT]; PLAYERS],
            crop_watered_today: [[false; TILE_COUNT]; PLAYERS],
            crop_unwatered: [[0; TILE_COUNT]; PLAYERS],
            crop_fertilized_until_day: [[-1; TILE_COUNT]; PLAYERS],
            seeds: [[0; CROP_TYPES]; PLAYERS],
            animal_fed: [[false; TILE_COUNT]; PLAYERS],
            animal_age: [[0; TILE_COUNT]; PLAYERS],
            animal_yield: [[0; TILE_COUNT]; PLAYERS],
            structure: [[false; TILE_COUNT]; PLAYERS],
            structure_kind: [[EMPTY_KIND; TILE_COUNT]; PLAYERS],
            animal_on_tile: [[false; TILE_COUNT]; PLAYERS],
            animal_kind: [[EMPTY_KIND; TILE_COUNT]; PLAYERS],
            animal_unfed: [[0; TILE_COUNT]; PLAYERS],
            animal_cared: [[false; TILE_COUNT]; PLAYERS],
            animal_fertilizer_available: [[false; TILE_COUNT]; PLAYERS],
            animal_pending_care: [[0; TILE_COUNT]; PLAYERS],
            weed: [[false; TILE_COUNT]; PLAYERS],
            shop_instances: [-1; MAX_SHOP_INSTANCES],
            shop_count: 0,
            shed: [[0; 12]; PLAYERS],
            shed_used: [0; PLAYERS],
            inventory: [[0; 9]; PLAYERS],
            animal_inventory: [[0; 3]; PLAYERS],
            hand_inventory: [[[0; 9]; MAX_HANDS]; PLAYERS],
            hand_animal_inventory: [[[0; 3]; MAX_HANDS]; PLAYERS],
            item_order_seq: 0,
            farmer_item_order: [[0; SHED_ITEMS]; PLAYERS],
            hand_item_order: [[[0; SHED_ITEMS]; MAX_HANDS]; PLAYERS],
            market_inventory: market_config.i0,
            market_prices: [0; PRODUCTS],
            market_dirty: (1_u16 << PRODUCTS) - 1,
            market_config,
            market_tables,
        };
        state.refresh_prices();
        #[cfg(debug_assertions)]
        state.debug_assert_invariants();
        state
    }

    fn market_price(&self, item: usize, inventory: f64) -> i32 {
        self.market_tables
            .get(item, inventory)
            .unwrap_or_else(|| market_price_reference(&self.market_config, item, inventory))
    }

    #[inline]
    fn market_price_integer(&self, item: usize, inventory: i32) -> i32 {
        self.market_tables
            .get_integer(item, inventory)
            .unwrap_or_else(|| market_price_reference(&self.market_config, item, inventory as f64))
    }

    fn shed_room(&self, player: usize) -> i32 {
        (self.shed_capacity as i32 - self.shed_used[player]).max(0)
    }

    #[cfg(debug_assertions)]
    fn debug_assert_invariants(&self) {
        for player in 0..PLAYERS {
            debug_assert!(self.money[player].is_finite());
            debug_assert!(self.money[player] >= 0.0);
            debug_assert!(self.hand_count[player] <= MAX_HANDS);
            debug_assert!(self.hires_today[player] <= MAX_HANDS);
            debug_assert!(self.shed_used[player] >= 0);
            debug_assert!(self.shed_used[player] <= self.shed_capacity as i32);

            let shed_total: i32 = self.shed[player].iter().sum();
            debug_assert_eq!(shed_total, self.shed_used[player]);
            for value in self.shed[player]
                .iter()
                .chain(self.inventory[player].iter())
                .chain(self.animal_inventory[player].iter())
            {
                debug_assert!(*value >= 0);
            }
            for hand in 0..MAX_HANDS {
                for value in self.hand_inventory[player][hand]
                    .iter()
                    .chain(self.hand_animal_inventory[player][hand].iter())
                {
                    debug_assert!(*value >= 0);
                }
            }

            let [x, y] = self.positions[player];
            debug_assert!((0..BOARD_SIZE as i32).contains(&x));
            debug_assert!((0..BOARD_SIZE as i32).contains(&y));
            for hand in 0..self.hand_count[player] {
                let [x, y] = self.hand_positions[player][hand];
                debug_assert!((0..BOARD_SIZE as i32).contains(&x));
                debug_assert!((0..BOARD_SIZE as i32).contains(&y));
            }

            for tile in 0..TILE_COUNT {
                let crop = self.crop_kind[player][tile];
                let has_crop = self.crop_age[player][tile] != EMPTY_CROP_AGE;
                debug_assert_eq!(has_crop, crop != EMPTY_KIND);
                if has_crop {
                    debug_assert!((0..CROP_TYPES as i8).contains(&crop));
                    debug_assert!(self.crop_yield[player][tile] >= 0);
                    debug_assert!(!self.weed[player][tile]);
                    debug_assert!(!self.structure[player][tile]);
                }
                if self.weed[player][tile] {
                    debug_assert!(!has_crop);
                    debug_assert!(!self.structure[player][tile]);
                }

                let structure_kind = self.structure_kind[player][tile];
                debug_assert_eq!(self.structure[player][tile], structure_kind != EMPTY_KIND);
                if self.structure[player][tile] {
                    debug_assert!((0..2).contains(&structure_kind));
                }
                let animal = self.animal_kind[player][tile];
                debug_assert_eq!(self.animal_on_tile[player][tile], animal != EMPTY_KIND);
                if self.animal_on_tile[player][tile] {
                    debug_assert!((0..ANIMAL_TYPES as i8).contains(&animal));
                    debug_assert!(self.structure[player][tile]);
                    debug_assert!(self.animal_yield[player][tile] >= 0);
                }
            }
        }
        for inventory in self.market_inventory {
            debug_assert!(inventory.is_finite());
            debug_assert!(inventory >= 0.0);
        }
        for price in self.market_prices {
            debug_assert!(price >= 1);
        }
        debug_assert!(self.shop_count <= self.shop_instances.len());
        for shop in self.shop_instances.iter().take(self.shop_count) {
            debug_assert!((0..SHOP_DEMANDS.len() as i8).contains(shop));
        }
    }

    fn refresh_prices(&mut self) {
        for item in 0..PRODUCTS {
            if self.market_dirty & (1 << item) != 0 {
                self.market_prices[item] = self.market_price(item, self.market_inventory[item]);
            }
        }
        self.market_dirty = 0;
    }

    #[inline]
    fn adjust_market_inventory(&mut self, item: usize, delta: f64) {
        self.market_inventory[item] += delta;
        self.market_dirty |= 1 << item;
    }

    fn apply_town_center(&mut self) {
        if self.step.is_multiple_of(self.center_interval) {
            for item in 0..8 {
                self.adjust_market_inventory(item, -1.0);
            }
        }
    }

    fn apply_town_shops(&mut self) {
        if !self.step.is_multiple_of(self.shop_sell_interval) {
            return;
        }
        for slot in 0..self.shop_count {
            let demands = SHOP_DEMANDS[self.shop_instances[slot] as usize];
            let multiplier = if demands[1] < 0 { 2 } else { 1 };
            for item in demands {
                if item >= 0 {
                    self.adjust_market_inventory(item as usize, -(multiplier as f64));
                }
            }
        }
    }

    fn hire_cost(&self, player: usize) -> i32 {
        let mut a = 1;
        let mut b = 1;
        for _ in 0..self.hires_today[player] {
            (a, b) = (b, a + b);
        }
        self.farm_hand_cost_mult * a
    }

    fn spawn_hand(&self, player: usize) -> [i32; 2] {
        let access = [[4, 4], [5, 4], [4, 5], [5, 5]];
        let mut occupancy = [0usize; 4];
        for position in [self.positions[player]].into_iter().chain(
            self.hand_positions[player][..self.hand_count[player]]
                .iter()
                .copied(),
        ) {
            for (index, tile) in access.iter().enumerate() {
                if position == *tile {
                    occupancy[index] += 1;
                }
            }
        }
        let mut best = 0;
        for index in 1..access.len() {
            if occupancy[index] < occupancy[best] {
                best = index;
            }
        }
        access[best]
    }

    fn tile_unlocked(&self, player: usize, tile: usize) -> bool {
        let x = tile % 10;
        let y = tile / 10;
        let quadrant = if y < 5 {
            if x < 5 {
                0
            } else {
                1
            }
        } else if x < 5 {
            2
        } else {
            3
        };
        self.unlocked[player][quadrant]
    }

    fn is_shed_adjacent_at(&self, position: [i32; 2]) -> bool {
        matches!(position, [4, 4] | [5, 4] | [4, 5] | [5, 5])
    }

    // Stamps `idx` with the next insertion-order sequence number iff this
    // call actually adds it to an empty slot (`current == 0 && delta > 0`),
    // mirroring a Python dict inserting a new key. Call BEFORE mutating
    // the underlying inventory/animal_inventory slot.
    fn mark_item_added(
        &mut self,
        order: &mut [u32; SHED_ITEMS],
        idx: usize,
        current: i32,
        delta: i32,
    ) {
        if delta > 0 && current == 0 {
            self.item_order_seq = self.item_order_seq.wrapping_add(1).max(1);
            order[idx] = self.item_order_seq;
        }
    }

    // Clears `idx`'s insertion-order stamp once its quantity reaches zero,
    // mirroring Python's `del inv[item]`. Call AFTER mutating.
    fn clear_item_if_empty(order: &mut [u32; SHED_ITEMS], idx: usize, new_value: i32) {
        if new_value <= 0 {
            order[idx] = 0;
        }
    }

    fn move_to_shed(&mut self, player: usize, item: usize, quantity: i32) -> i32 {
        let moved = quantity.max(0).min(self.shed_room(player));
        self.shed[player][item] += moved;
        self.shed_used[player] += moved;
        moved
    }

    fn move_from_shed(&mut self, player: usize, item: usize, quantity: i32) -> i32 {
        let moved = quantity.max(0).min(self.shed[player][item]);
        self.shed[player][item] -= moved;
        self.shed_used[player] -= moved;
        moved
    }

    fn apply_unit_action(
        &mut self,
        player: usize,
        unit: &mut ActiveUnit,
        operation: i64,
        target: i64,
        quantity: i64,
    ) {
        let tile = tile_index(unit.position);
        if operation != 0
            && !(1..=4).contains(&operation)
            && !matches!(operation, 9 | 11 | 15)
            && !self.tile_unlocked(player, tile)
        {
            return;
        }
        if operation == 11 && !self.is_shed_adjacent_at(unit.position) {
            return;
        }
        match operation {
            1..=4 => move_position(&mut unit.position, operation),
            5 if (0..5).contains(&target) => {
                let crop = target as usize;
                if self.crop_age[player][tile] == EMPTY_CROP_AGE
                    && !self.structure[player][tile]
                    && !self.weed[player][tile]
                    && self.seeds[player][crop] > 0
                {
                    self.crop_age[player][tile] = 0;
                    self.crop_kind[player][tile] = target as i8;
                    let planted_day = self.step / self.turns_per_day;
                    let decay_day = if CROP_INTERVAL[crop] == 0 {
                        planted_day + CROP_MAX_YIELD_DAY[crop] as u32 + 1
                    } else {
                        planted_day
                            + CROP_FIRST_YIELD_DAY[crop] as u32
                            + ((CROP_MAX_YIELD[crop] - 1) * CROP_INTERVAL[crop]) as u32
                            + 1
                    };
                    self.crop_decay_step[player][tile] = (decay_day * self.turns_per_day) as i32;
                    self.crop_yield[player][tile] = if CROP_INTERVAL[crop] == 0 { 1 } else { 0 };
                    self.crop_unwatered[player][tile] = 1;
                    self.weed[player][tile] = false;
                    self.seeds[player][crop] -= 1;
                    // A prior crop's fertilizer bonus must not carry over to
                    // this new planting. Python's `_new_plant` always builds
                    // a brand-new tile dict with `"fertilized_until_day": -1`
                    // (kaggriculture.py); without this reset, replanting on a
                    // tile that was fertilized under an earlier crop (then
                    // harvested, decayed, or dug) silently granted the new,
                    // never-fertilized crop a watering yield bonus until the
                    // stale day threshold happened to pass.
                    self.crop_fertilized_until_day[player][tile] = -1;
                    // Same leak, different field: `_new_plant` also always
                    // sets `"watered_today": False`. A single-harvest crop
                    // (CROP_INTERVAL == 0) that was WATERed today, then
                    // HARVESTed (which clears crop_age/crop_kind but not
                    // crop_watered_today -- see op 6 below), left this tile
                    // stuck at "already watered today" for whatever gets
                    // planted here next this same day, silently blocking
                    // that crop's own first WATER of the day (and its yield
                    // bonus) even though Python's fresh tile dict allows it.
                    self.crop_watered_today[player][tile] = false;
                }
            }
            6 => {
                if self.animal_on_tile[player][tile] {
                    let animal = self.animal_kind[player][tile] as usize;
                    if animal < 3 && self.animal_yield[player][tile] > 0 {
                        // One HARVEST drains all accumulated yield at once,
                        // matching the crop branch just below and the
                        // Python engine's animal HARVEST (`units =
                        // tile["yield_units"]; tile["yield_units"] = 0`).
                        let product = ANIMAL_PRODUCT[animal];
                        let amount = self.animal_yield[player][tile];
                        self.mark_item_added(
                            &mut unit.item_order,
                            product,
                            unit.inventory[product],
                            amount,
                        );
                        unit.inventory[product] += amount;
                        self.animal_yield[player][tile] = 0;
                    }
                } else if self.crop_age[player][tile] != EMPTY_CROP_AGE
                    && self.crop_yield[player][tile] > 0
                    && self.crop_age[player][tile]
                        >= CROP_FIRST_YIELD_DAY[self.crop_kind[player][tile] as usize]
                {
                    let crop = self.crop_kind[player][tile] as usize;
                    let amount = self.crop_yield[player][tile];
                    self.mark_item_added(&mut unit.item_order, crop, unit.inventory[crop], amount);
                    unit.inventory[crop] += amount;
                    self.crop_yield[player][tile] = 0;
                    // Single-harvest crops (CROP_INTERVAL == 0) are removed from
                    // the tile. Ongoing crops keep growing after their yield is
                    // collected, matching the Python engine's `ongoing` flag.
                    if CROP_INTERVAL[crop] == 0 {
                        self.crop_age[player][tile] = EMPTY_CROP_AGE;
                        self.crop_kind[player][tile] = EMPTY_KIND;
                        self.crop_decay_step[player][tile] = -1;
                    }
                }
            }
            7 if self.animal_on_tile[player][tile] => {
                if !self.animal_fed[player][tile] && unit.inventory[0] > 0 {
                    unit.inventory[0] -= 1;
                    GameState::clear_item_if_empty(&mut unit.item_order, 0, unit.inventory[0]);
                    self.animal_fed[player][tile] = true;
                }
            }
            10 => {
                let age = self.crop_age[player][tile];
                if age >= 0 && !self.crop_watered_today[player][tile] {
                    let crop = self.crop_kind[player][tile] as usize;
                    // The watering yield bonus applies only to single-harvest
                    // crops. Ongoing crops (CROP_INTERVAL > 0) accrue yield
                    // through the daily interval refresh instead.
                    if CROP_INTERVAL[crop] == 0 {
                        let window_start = (CROP_MAX_YIELD_DAY[crop] + 1) / 2;
                        if age >= window_start && age <= CROP_MAX_YIELD_DAY[crop] {
                            let day = self.step as i32 / self.turns_per_day as i32;
                            let bonus = if self.crop_fertilized_until_day[player][tile] >= day {
                                2
                            } else {
                                1
                            };
                            self.crop_yield[player][tile] =
                                (self.crop_yield[player][tile] + bonus).min(CROP_MAX_YIELD[crop]);
                        }
                    }
                }
                if age >= 0 {
                    self.crop_watered_today[player][tile] = true;
                }
            }
            8 if (target == 0 || target == 1)
                && self.crop_age[player][tile] == EMPTY_CROP_AGE
                && !self.weed[player][tile]
                && !self.structure[player][tile] =>
            {
                self.structure[player][tile] = true;
                self.structure_kind[player][tile] = target as i8;
            }
            9 if (0..3).contains(&target) => {
                let animal = target as usize;
                if self.structure[player][tile]
                    && !self.animal_on_tile[player][tile]
                    && unit.animal_inventory[animal] > 0
                    && self.structure_kind[player][tile] == ANIMAL_STRUCTURE[animal]
                {
                    unit.animal_inventory[animal] -= 1;
                    GameState::clear_item_if_empty(
                        &mut unit.item_order,
                        PRODUCTS + animal,
                        unit.animal_inventory[animal],
                    );
                    self.animal_on_tile[player][tile] = true;
                    self.animal_kind[player][tile] = target as i8;
                    self.animal_age[player][tile] = 0;
                    self.animal_yield[player][tile] = 0;
                    self.animal_unfed[player][tile] = 0;
                    self.animal_fed[player][tile] = false;
                    self.animal_cared[player][tile] = false;
                    self.animal_pending_care[player][tile] = 0;
                    self.animal_fertilizer_available[player][tile] = false;
                } else if self.is_shed_adjacent_at(unit.position) {
                    // Official PLACE shed fallback: n <= 0 is a silent no-op
                    // (`if n <= 0 { return; }` in kaggriculture.py), and the
                    // requested count is unbounded -- availability and shed
                    // room are the only effective bounds.
                    if quantity <= 0 {
                        return;
                    }
                    let room = self.shed_room(player);
                    let requested = quantity.clamp(1, i64::from(i32::MAX)) as i32;
                    let available = unit.animal_inventory[animal];
                    let moved =
                        self.move_to_shed(player, 9 + animal, available.min(room).min(requested));
                    unit.animal_inventory[animal] -= moved;
                    GameState::clear_item_if_empty(
                        &mut unit.item_order,
                        PRODUCTS + animal,
                        unit.animal_inventory[animal],
                    );
                }
            }
            9 if (4..=12).contains(&target) && self.is_shed_adjacent_at(unit.position) => {
                // Official PLACE shed path: n <= 0 is a silent no-op.
                if quantity <= 0 {
                    return;
                }
                let item = (target - 4) as usize;
                let room = self.shed_room(player);
                let requested = quantity.clamp(1, i64::from(i32::MAX)) as i32;
                let available = unit.inventory[item];
                let moved = self.move_to_shed(player, item, available.min(room).min(requested));
                unit.inventory[item] -= moved;
                GameState::clear_item_if_empty(&mut unit.item_order, item, unit.inventory[item]);
            }
            11 if target == 0 => {
                if quantity <= 0 {
                    return;
                }
                // Official PICKUP n is unbounded; shed stock is the only
                // effective bound (`n = min(n, available)`).
                let quantity = quantity.clamp(1, i64::from(i32::MAX)) as i32;
                let moved = self.move_from_shed(player, 0, quantity);
                self.mark_item_added(&mut unit.item_order, 0, unit.inventory[0], moved);
                unit.inventory[0] += moved;
            }
            11 if (1..=3).contains(&target) => {
                if quantity <= 0 {
                    return;
                }
                let animal = (target - 1) as usize;
                let quantity = quantity.clamp(1, i64::from(i32::MAX)) as i32;
                let moved = self.move_from_shed(player, 9 + animal, quantity);
                self.mark_item_added(
                    &mut unit.item_order,
                    PRODUCTS + animal,
                    unit.animal_inventory[animal],
                    moved,
                );
                unit.animal_inventory[animal] += moved;
            }
            11 if (4..=12).contains(&target) => {
                if quantity <= 0 {
                    return;
                }
                let item = (target - 4) as usize;
                let quantity = quantity.clamp(1, i64::from(i32::MAX)) as i32;
                let moved = self.move_from_shed(player, item, quantity);
                self.mark_item_added(&mut unit.item_order, item, unit.inventory[item], moved);
                unit.inventory[item] += moved;
            }
            12 if self.crop_age[player][tile] != EMPTY_CROP_AGE && unit.inventory[8] > 0 => {
                unit.inventory[8] -= 1;
                GameState::clear_item_if_empty(&mut unit.item_order, 8, unit.inventory[8]);
                let day = self.step as i32 / self.turns_per_day as i32;
                self.crop_fertilized_until_day[player][tile] =
                    self.crop_fertilized_until_day[player][tile].max(day + 2);
            }
            13 if self.animal_on_tile[player][tile] => {
                self.animal_cared[player][tile] = true;
            }
            14 if self.animal_on_tile[player][tile] => {
                if self.animal_fertilizer_available[player][tile] {
                    self.animal_fertilizer_available[player][tile] = false;
                    self.mark_item_added(&mut unit.item_order, 8, unit.inventory[8], 1);
                    unit.inventory[8] += 1;
                }
            }
            15 if self.is_shed_adjacent_at(unit.position) => {
                // Python's DROP (kaggriculture.py) unconditionally clears
                // the whole carried inventory after computing how much fits
                // in the shed (`del inv[item]` runs regardless of `take`),
                // discarding whatever doesn't fit rather than leaving it in
                // the farmer's hands. It also iterates the farmer's
                // inventory `dict` in INSERTION order, recomputing shed
                // room fresh each step, so whichever item entered the dict
                // first gets priority for scarce shed room -- a fixed
                // products-then-animals order (as this used to iterate)
                // gives priority to the wrong item whenever a unit carries
                // 2+ item types and the shed can't fit all of them. Process
                // items by `unit.item_order` (ascending insertion stamp) to
                // match; items with no stamp (order == 0) but a positive
                // quantity are a defensive fallback and sort first.
                let mut order: Vec<usize> = (0..SHED_ITEMS)
                    .filter(|&idx| {
                        if idx < PRODUCTS {
                            unit.inventory[idx] > 0
                        } else {
                            unit.animal_inventory[idx - PRODUCTS] > 0
                        }
                    })
                    .collect();
                order.sort_by_key(|&idx| unit.item_order[idx]);
                for idx in order {
                    let room = self.shed_room(player);
                    if idx < PRODUCTS {
                        self.move_to_shed(player, idx, unit.inventory[idx].min(room));
                        unit.inventory[idx] = 0;
                    } else {
                        let animal = idx - PRODUCTS;
                        self.move_to_shed(player, idx, unit.animal_inventory[animal].min(room));
                        unit.animal_inventory[animal] = 0;
                    }
                    unit.item_order[idx] = 0;
                }
            }
            17 => {
                if self.crop_age[player][tile] != EMPTY_CROP_AGE {
                    self.crop_age[player][tile] = EMPTY_CROP_AGE;
                    self.crop_kind[player][tile] = EMPTY_KIND;
                    self.crop_decay_step[player][tile] = -1;
                    self.crop_yield[player][tile] = 0;
                    self.crop_watered_today[player][tile] = false;
                    self.crop_unwatered[player][tile] = 0;
                    self.crop_fertilized_until_day[player][tile] = -1;
                }
                self.weed[player][tile] = false;
                if !self.animal_on_tile[player][tile] {
                    self.structure[player][tile] = false;
                    self.structure_kind[player][tile] = EMPTY_KIND;
                }
            }
            _ => {}
        }
    }

    fn apply_hand_action(
        &mut self,
        player: usize,
        hand: usize,
        operation: i64,
        target: i64,
        quantity: i64,
    ) {
        if hand >= self.hand_count[player] {
            return;
        }
        if operation == 0 {
            return;
        }
        if (1..=4).contains(&operation) {
            move_position(&mut self.hand_positions[player][hand], operation);
            return;
        }
        let mut unit = ActiveUnit {
            position: self.hand_positions[player][hand],
            inventory: self.hand_inventory[player][hand],
            animal_inventory: self.hand_animal_inventory[player][hand],
            item_order: self.hand_item_order[player][hand],
        };
        self.apply_unit_action(player, &mut unit, operation, target, quantity);
        self.hand_positions[player][hand] = unit.position;
        self.hand_inventory[player][hand] = unit.inventory;
        self.hand_animal_inventory[player][hand] = unit.animal_inventory;
        self.hand_item_order[player][hand] = unit.item_order;
    }

    fn apply_market_action(
        &mut self,
        player: usize,
        operation: i64,
        target: i64,
        quantity: i64,
        quoted_price: Option<i32>,
    ) {
        // Official order quantities are unbounded (`_parse_order` accepts any
        // positive int); funds, shed room, and stock are the only bounds.
        let quantity = quantity.clamp(0, i64::from(i32::MAX)) as i32;
        if quantity == 0 {
            return;
        }
        match operation {
            1 if (0..5).contains(&target) => {
                let crop = target as usize;
                // i64 product: seed cost * an unbounded quantity would
                // overflow i32 (official Python uses bigints and simply
                // fails the money check).
                let cost = i64::from(CROP_SEED_COSTS[crop]) * i64::from(quantity);
                if self.money[player] + MONEY_EPSILON >= cost as f32 {
                    self.money[player] = (self.money[player] - cost as f32).max(0.0);
                    self.seeds[player][crop] += quantity;
                }
            }
            2 if (target == 0 || target == 8)
                && self.shed_used[player] + quantity <= self.shed_capacity as i32
                && self.money[player]
                    >= quoted_price.unwrap_or_else(|| {
                        self.market_price_integer(
                            target as usize,
                            self.market_inventory[target as usize] as i32 - 1,
                        )
                    }) as f32
                        * quantity as f32 =>
            {
                let item = target as usize;
                let cost = quoted_price.unwrap_or_else(|| {
                    self.market_price_integer(item, self.market_inventory[item] as i32 - 1)
                }) * quantity;
                self.money[player] -= cost as f32;
                self.move_to_shed(player, item, quantity);
                self.adjust_market_inventory(item, -(quantity as f64));
            }
            3 if (0..3).contains(&target)
                && self.shed_used[player] + quantity <= self.shed_capacity as i32
                && self.money[player] >= ANIMAL_COSTS[target as usize] as f32 * quantity as f32 =>
            {
                let animal = target as usize;
                self.money[player] -= ANIMAL_COSTS[animal] as f32 * quantity as f32;
                self.move_to_shed(player, 9 + animal, quantity);
            }
            4 if (0..9).contains(&target) && self.shed[player][target as usize] >= quantity => {
                let item = target as usize;
                self.move_from_shed(player, item, quantity);
                let price = quoted_price.unwrap_or(self.market_prices[item]);
                self.money[player] += price as f32 * quantity as f32;
                if price > 1 {
                    self.adjust_market_inventory(item, quantity as f64);
                }
            }
            5 if self.hand_count[player] < MAX_HANDS => {
                let cost = self.hire_cost(player);
                if self.money[player] + MONEY_EPSILON >= cost as f32 {
                    let hand = self.spawn_hand(player);
                    self.money[player] = (self.money[player] - cost as f32).max(0.0);
                    self.hand_positions[player][self.hand_count[player]] = hand;
                    self.hand_count[player] += 1;
                    self.hires_today[player] += 1;
                }
            }
            6 => {
                let land = self.unlocked[player].iter().skip(1).position(|open| !open);
                if let Some(index) = land {
                    let cost = LAND_PRICES[index];
                    if self.money[player] >= cost as f32 {
                        self.money[player] -= cost as f32;
                        self.unlocked[player][index + 1] = true;
                    }
                }
            }
            _ => {}
        }
    }

    // Bulk processing applies only when the other player cannot act.
    // The interleaved path remains the reference path for two active players.
    fn apply_market_bulk(&mut self, player: usize, operation: i64, target: i64, quantity: i64) {
        // Unbounded like the official order quantity; every branch below
        // bounds its work by shed room or stock.
        let quantity = quantity.clamp(0, i64::from(i32::MAX)) as i32;
        if quantity == 0 {
            return;
        }
        match operation {
            2 if target == 0 || target == 8 => {
                let item = target as usize;
                let room = self.shed_room(player).min(quantity);
                let mut accepted = 0;
                let mut total_cost = 0.0_f32;
                if let Some(max_cost) = self.market_tables.buy_cost_integer(
                    item,
                    self.market_inventory[item] as i32,
                    room,
                ) {
                    let mut low = 0;
                    let mut high = room;
                    while low < high {
                        let candidate = low + (high - low + 1) / 2;
                        let cost = self
                            .market_tables
                            .buy_cost_integer(item, self.market_inventory[item] as i32, candidate)
                            .expect("validated market lookup range");
                        if (cost as f32) <= self.money[player] {
                            low = candidate;
                        } else {
                            high = candidate - 1;
                        }
                    }
                    accepted = low;
                    total_cost = if accepted == room {
                        max_cost as f32
                    } else {
                        self.market_tables
                            .buy_cost_integer(item, self.market_inventory[item] as i32, accepted)
                            .expect("validated market lookup range") as f32
                    };
                } else {
                    let mut inventory = self.market_inventory[item] as i32;
                    while accepted < room {
                        let price = self.market_price_integer(item, inventory - 1) as f32;
                        if self.money[player] - total_cost < price {
                            break;
                        }
                        total_cost += price;
                        inventory -= 1;
                        accepted += 1;
                    }
                }
                if accepted > 0 {
                    self.money[player] -= total_cost;
                    self.move_to_shed(player, item, accepted);
                    self.adjust_market_inventory(item, -(accepted as f64));
                }
            }
            4 if (0..9).contains(&target) => {
                let item = target as usize;
                let accepted = quantity.min(self.shed[player][item]);
                let (total_revenue, market_supply_increase) = self
                    .market_tables
                    .sell_totals_integer(item, self.market_inventory[item] as i32, accepted)
                    .map(|(revenue, supply)| (revenue as f32, supply))
                    .unwrap_or_else(|| {
                        let mut total_revenue = 0.0_f32;
                        let mut inventory = self.market_inventory[item] as i32;
                        let mut market_supply_increase = 0;
                        for _ in 0..accepted {
                            let price = self.market_price_integer(item, inventory);
                            total_revenue += price as f32;
                            if price > 1 {
                                inventory += 1;
                                market_supply_increase += 1;
                            }
                        }
                        (total_revenue, market_supply_increase)
                    });
                if accepted > 0 {
                    self.move_from_shed(player, item, accepted);
                    self.money[player] += total_revenue;
                    self.adjust_market_inventory(item, market_supply_increase as f64);
                }
            }
            _ => {}
        }
    }

    fn market_action_possible(
        &self,
        player: usize,
        operation: i64,
        target: i64,
        quoted_price: Option<i32>,
    ) -> bool {
        match operation {
            1 if (0..5).contains(&target) => {
                let cost = CROP_SEED_COSTS[target as usize];
                self.money[player] >= cost as f32
            }
            2 if target == 0 || target == 8 => {
                let price = quoted_price.unwrap_or_else(|| {
                    self.market_price_integer(
                        target as usize,
                        self.market_inventory[target as usize] as i32 - 1,
                    )
                });
                self.shed_used[player] < self.shed_capacity as i32
                    && self.money[player] >= price as f32
            }
            3 if (0..3).contains(&target) => {
                self.shed_used[player] < self.shed_capacity as i32
                    && self.money[player] >= ANIMAL_COSTS[target as usize] as f32
            }
            4 if (0..9).contains(&target) => self.shed[player][target as usize] > 0,
            _ => false,
        }
    }

    /// Apply one farm's end-of-day crop/animal growth, decay-to-weed,
    /// yield accrual, shed transfer, and hand reset -- everything
    /// `advance_crops_and_animals` (production, called every turn) does
    /// at a day boundary except weed-spawning. Weed-spawning stays in
    /// the caller rather than moving here too, since it needs the
    /// caller's shared per-day RNG stream at a specific position in that
    /// stream (immediately after this call, per player, matching
    /// production's exact draw order) -- folding it into this method
    /// would require threading the RNG through a method whose whole
    /// point is to not need one.
    ///
    /// This method applies growth, decay, yield transfer, and hand reset.
    /// Weed spawning stays in the production transition that owns its RNG order.
    fn advance_farm_growth(&mut self, player: usize) {
        for age in self.crop_age[player].iter_mut() {
            if *age >= 0 {
                *age += 1;
            }
        }
        for tile in 0..TILE_COUNT {
            if !self.animal_on_tile[player][tile] {
                continue;
            }
            let animal = self.animal_kind[player][tile] as usize;
            if animal >= 3 {
                continue;
            }
            if self.animal_fed[player][tile] {
                self.animal_unfed[player][tile] = 0;
            } else {
                self.animal_unfed[player][tile] += 1;
            }
            if self.animal_unfed[player][tile] >= 2 {
                self.animal_on_tile[player][tile] = false;
                self.animal_kind[player][tile] = EMPTY_KIND;
                continue;
            }
            self.animal_age[player][tile] += 1;
            let age = self.animal_age[player][tile];
            // Base yield (+1) accrues on schedule regardless of feeding --
            // only the pending-care bonus requires the animal to have been
            // fed on the production day. Feeding only prevents starvation
            // (2 consecutive unfed days above) and unlocks the bonus; it is
            // not a precondition for base production. Matches the Python
            // engine's `_daily_refresh_animals` exactly: `base = 1` is
            // unconditional, `bonus = tile.pop(...) if tile["fed_today"]
            // else 0`, and `pending_care_bonus` resets to 0 either way once
            // the interval matches.
            if age >= ANIMAL_FIRST_YIELD_DAY[animal]
                && (age - ANIMAL_FIRST_YIELD_DAY[animal]) % ANIMAL_INTERVAL[animal] == 0
            {
                let bonus = if self.animal_fed[player][tile] {
                    self.animal_pending_care[player][tile]
                } else {
                    0
                };
                self.animal_pending_care[player][tile] = 0;
                self.animal_yield[player][tile] =
                    (self.animal_yield[player][tile] + 1 + bonus).min(ANIMAL_MAX_HELD[animal]);
            }
            if self.animal_fed[player][tile] && self.animal_cared[player][tile] {
                self.animal_pending_care[player][tile] += 1;
            }
            self.animal_fertilizer_available[player][tile] = true;
            self.animal_cared[player][tile] = false;
            self.animal_fed[player][tile] = false;
        }
        for tile in 0..TILE_COUNT {
            let crop = self.crop_kind[player][tile];
            if crop < 0 {
                continue;
            }
            let crop = crop as usize;
            let age = self.crop_age[player][tile];
            if self.crop_watered_today[player][tile] {
                self.crop_unwatered[player][tile] = 0;
            } else {
                self.crop_unwatered[player][tile] += 1;
            }
            if self.crop_unwatered[player][tile] >= 2 {
                self.crop_age[player][tile] = EMPTY_CROP_AGE;
                self.crop_kind[player][tile] = EMPTY_KIND;
                self.crop_yield[player][tile] = 0;
                self.crop_decay_step[player][tile] = -1;
                self.weed[player][tile] = true;
                continue;
            }
            if CROP_INTERVAL[crop] > 0 {
                let days_since_first = age - CROP_FIRST_YIELD_DAY[crop];
                if days_since_first >= 0 && days_since_first % CROP_INTERVAL[crop] == 0 {
                    let count = days_since_first / CROP_INTERVAL[crop] + 1;
                    if count <= CROP_MAX_YIELD[crop] {
                        let day = (self.step + 1) as i32 / self.turns_per_day as i32;
                        let bonus = if self.crop_watered_today[player][tile]
                            && self.crop_fertilized_until_day[player][tile] >= day - 1
                        {
                            2
                        } else {
                            1
                        };
                        self.crop_yield[player][tile] =
                            (self.crop_yield[player][tile] + bonus).min(CROP_MAX_YIELD[crop]);
                    }
                }
            }
        }
        self.crop_watered_today[player].fill(false);
        // Python's end-of-day `_drop_inventories_to_shed` iterates
        // `private["inventories"]` (farmer, then hands in index order) and,
        // within each entity's own inventory `dict`, its insertion order --
        // not a fixed products-then-animals array order. Same divergence as
        // the op-15 DROP fix above (see finding 11): use each entity's
        // `item_order` stamps so whichever item was picked up first within
        // that entity gets priority for the shed room that's left once
        // earlier entities (farmer first) have already taken theirs.
        self.drop_entity_inventory_to_shed(player, EntityInventoryRef::Farmer);
        for hand in 0..self.hand_count[player] {
            self.drop_entity_inventory_to_shed(player, EntityInventoryRef::Hand(hand));
        }
        self.positions[player] = [4, 4];
        self.hand_count[player] = 0;
        self.hires_today[player] = 0;
        self.hand_positions[player] = [[0, 0]; MAX_HANDS];
    }

    fn drop_entity_inventory_to_shed(&mut self, player: usize, entity: EntityInventoryRef) {
        let (inventory, animal_inventory, order) = match entity {
            EntityInventoryRef::Farmer => (
                self.inventory[player],
                self.animal_inventory[player],
                self.farmer_item_order[player],
            ),
            EntityInventoryRef::Hand(hand) => (
                self.hand_inventory[player][hand],
                self.hand_animal_inventory[player][hand],
                self.hand_item_order[player][hand],
            ),
        };
        let mut indices: Vec<usize> = (0..SHED_ITEMS)
            .filter(|&idx| {
                if idx < PRODUCTS {
                    inventory[idx] > 0
                } else {
                    animal_inventory[idx - PRODUCTS] > 0
                }
            })
            .collect();
        indices.sort_by_key(|&idx| order[idx]);
        for idx in indices {
            let room = self.shed_room(player);
            let amount = if idx < PRODUCTS {
                inventory[idx]
            } else {
                animal_inventory[idx - PRODUCTS]
            };
            self.move_to_shed(player, idx, amount.min(room));
        }
        match entity {
            EntityInventoryRef::Farmer => {
                self.inventory[player] = [0; 9];
                self.animal_inventory[player] = [0; ANIMAL_TYPES];
                self.farmer_item_order[player] = [0; SHED_ITEMS];
            }
            EntityInventoryRef::Hand(hand) => {
                self.hand_inventory[player][hand] = [0; 9];
                self.hand_animal_inventory[player][hand] = [0; ANIMAL_TYPES];
                self.hand_item_order[player][hand] = [0; SHED_ITEMS];
            }
        }
    }

    fn advance_crops_and_animals(&mut self) {
        let end_of_day = (self.step + 1).is_multiple_of(self.turns_per_day);
        let day = self.step / self.turns_per_day;
        let mut rng = end_of_day.then(|| {
            PythonRandom::new(((self.episode_seed as u128) * 1_000_003_u128) ^ day as u128)
        });
        for player in 0..PLAYERS {
            if end_of_day {
                self.advance_farm_growth(player);
                for tile in 0..TILE_COUNT {
                    if self.crop_age[player][tile] == EMPTY_CROP_AGE
                        && !self.structure[player][tile]
                        && !self.weed[player][tile]
                        && self.tile_unlocked(player, tile)
                        && rng.as_mut().unwrap().random() < self.weed_spawn_chance
                    {
                        self.weed[player][tile] = true;
                    }
                }
            }
        }
        let next_day = (self.step + 1) / self.turns_per_day;
        if end_of_day
            && next_day > 0
            && next_day.is_multiple_of(self.shop_unlock_interval_days)
            && self.shop_count < MAX_SHOP_INSTANCES
        {
            self.shop_instances[self.shop_count] =
                rng.as_mut().unwrap().choice(SHOP_DEMANDS.len()) as i8;
            self.shop_count += 1;
        }
    }

    /// Step at which a planted crop starts its post-production decay
    /// countdown, or `None` if it has not (yet) reached that point.
    ///
    /// Single-harvest crops (CROP_INTERVAL == 0) decay on a fixed schedule
    /// from `max_yield_day`. Ongoing crops (CROP_INTERVAL > 0) only start
    /// decaying once their daily-refresh production count has reached
    /// `max_yield`, matching the Python engine's `max_lifespan_step`, which
    /// starts at -1 and is set the day production finishes.
    fn crop_lifespan_step(&self, player: usize, tile: usize) -> Option<u32> {
        let crop = self.crop_kind[player][tile];
        if crop < 0 {
            return None;
        }
        let crop = crop as usize;
        let lifespan = self.crop_decay_step[player][tile];
        if lifespan < 0 {
            return None;
        }
        if CROP_INTERVAL[crop] > 0 {
            let final_production_day = lifespan as u32 / self.turns_per_day - 1;
            if self.step / self.turns_per_day < final_production_day {
                return None;
            }
        }
        Some(lifespan as u32)
    }

    fn decay_plants(&mut self) {
        for player in 0..PLAYERS {
            for tile in 0..TILE_COUNT {
                let Some(lifespan) = self.crop_lifespan_step(player, tile) else {
                    continue;
                };
                if self.step >= lifespan && (self.step - lifespan).is_multiple_of(2) {
                    // Python's `_decay_plants` decrements `yield_units`
                    // UNCONDITIONALLY once the schedule fires and converts the
                    // tile to WEED when the result is <= 0. Gating on
                    // `yield > 0` (as this used to) skipped an ongoing crop
                    // whose yield was harvested down to 0 exactly when its
                    // production completed: Python turned that depleted plant
                    // into a WEED at `max_lifespan_step`, this kept it alive
                    // as a zero-yield PLANT forever.
                    self.crop_yield[player][tile] -= 1;
                    if self.crop_yield[player][tile] <= 0 {
                        self.crop_age[player][tile] = EMPTY_CROP_AGE;
                        self.crop_kind[player][tile] = EMPTY_KIND;
                        self.crop_decay_step[player][tile] = -1;
                        self.weed[player][tile] = true;
                    }
                }
            }
        }
    }
    fn observe(&self, player: usize, destination: &mut [f32]) {
        destination.fill(0.0);
        destination[0] = self.step as f32 / SEASON_STEPS;
        destination[1] = ((self.step / self.turns_per_day) as f32 / SEASON_DAYS).min(1.0);
        destination[2] = (self.step % self.turns_per_day) as f32 / TURNS_PER_DAY_NORMALIZED;
        destination[3] = player as f32;
        destination[4] = OBS_REMAINING_OVERAGE_NORMALIZED;
        destination[5] = self.money[0] / NORMALIZE_MONEY;
        destination[6] = self.money[1] / NORMALIZE_MONEY;
        for farm in 0..PLAYERS {
            let position_offset = 7 + farm * 6;
            destination[position_offset] = 1.0;
            destination[position_offset + 1] =
                self.positions[farm][0] as f32 / NORMALIZE_COORDINATES;
            destination[position_offset + 2] =
                self.positions[farm][1] as f32 / NORMALIZE_COORDINATES;
            // The reference packed observation leaves three farm-header
            // fields reserved.  Preserve the existing layout and expose the
            // public hires_today counter in the first reserved field so the
            // narrow Python API remains compatible with current agents.
            destination[position_offset + 3] = self.hires_today[farm] as f32 / MAX_HANDS as f32;
            let land_offset = 19 + farm * 4;
            for quadrant in 0..4 {
                destination[land_offset + quadrant] = self.unlocked[farm][quadrant] as u8 as f32;
            }
        }
        for farm in 0..PLAYERS {
            let farm_tile_offset = OBS_FARM_BASE + farm * generated_protocol::OBS_FARM_WIDTH;
            for tile in 0..TILE_COUNT {
                let tile_offset = farm_tile_offset + tile * generated_protocol::OBS_TILE_WIDTH;
                if !self.tile_unlocked(farm, tile) {
                    destination[tile_offset + 1] = 1.0;
                } else {
                    destination[tile_offset] = 1.0;
                }
                if self.weed[farm][tile] {
                    destination[tile_offset] = 0.0;
                    destination[tile_offset + 3] = 1.0;
                } else if self.crop_age[farm][tile] != EMPTY_CROP_AGE {
                    destination[tile_offset] = 0.0;
                    destination[tile_offset + 2] = 1.0;
                    let crop = self.crop_kind[farm][tile] as usize;
                    destination[tile_offset + 7 + crop] = 1.0;
                    destination[tile_offset + 6] = 1.0;
                    destination[tile_offset + 14] =
                        (self.crop_age[farm][tile] as f32 / SEASON_DAYS).min(1.0);
                    destination[tile_offset + 15] =
                        self.crop_yield[farm][tile] as f32 / NORMALIZE_QUANTITY;
                    destination[tile_offset + 16] = match self.crop_lifespan_step(farm, tile) {
                        Some(lifespan) => lifespan as f32 / SEASON_STEPS,
                        None => -1.0 / SEASON_STEPS,
                    };
                    destination[tile_offset + 17] =
                        self.crop_watered_today[farm][tile] as u8 as f32;
                    destination[tile_offset + 18] = self.crop_unwatered[farm][tile] as f32 / 2.0;
                    destination[tile_offset + 19] =
                        self.crop_fertilized_until_day[farm][tile] as f32 / SEASON_DAYS;
                } else if self.structure[farm][tile] {
                    destination[tile_offset] = 0.0;
                    destination[tile_offset + 4] = 1.0;
                    if self.structure_kind[farm][tile] == 1 {
                        destination[tile_offset + 4] = 0.0;
                        destination[tile_offset + 5] = 1.0;
                    }
                    if self.animal_on_tile[farm][tile] {
                        let animal = self.animal_kind[farm][tile] as usize;
                        destination[tile_offset + 11] = 1.0;
                        destination[tile_offset + 12 + animal] = 1.0;
                        destination[tile_offset + 15] =
                            (self.animal_yield[farm][tile] as f32 / NORMALIZE_QUANTITY).min(1.0);
                        destination[tile_offset + 20] =
                            self.animal_age[farm][tile] as f32 / SEASON_DAYS;
                        destination[tile_offset + 21] = self.animal_fed[farm][tile] as u8 as f32;
                        destination[tile_offset + 22] = self.animal_unfed[farm][tile] as f32 / 2.0;
                        destination[tile_offset + 23] = self.animal_cared[farm][tile] as u8 as f32;
                        destination[tile_offset + 24] =
                            (self.animal_fertilizer_available[farm][tile] as u8 as f32
                                / NORMALIZE_QUANTITY)
                                .min(1.0);
                        destination[tile_offset + 25] =
                            (self.animal_pending_care[farm][tile] as f32 / NORMALIZE_QUANTITY)
                                .min(1.0);
                    }
                }
            }
        }
        let market_offset = OBS_MARKET_INVENTORY;
        for (index, value) in self.market_inventory.iter().enumerate() {
            destination[market_offset + index] = *value as f32 / MARKET_I0[index] as f32;
        }
        let price_offset = OBS_MARKET_PRICES;
        for (index, value) in self.market_prices.iter().enumerate() {
            destination[price_offset + index] = *value as f32 / NORMALIZE_PRICE;
        }
        for slot in 0..self.shop_count {
            destination[OBS_SHOPS + slot] = (self.shop_instances[slot] as f32 + 1.0) / 8.0;
        }
        for (index, value) in self.shed[player].iter().enumerate() {
            destination[OBS_SHED + index] = *value as f32 / NORMALIZE_QUANTITY;
        }
        for (index, value) in self.seeds[player].iter().enumerate() {
            destination[OBS_SEEDS + index] = *value as f32 / NORMALIZE_QUANTITY;
        }
        for (index, value) in self.inventory[player].iter().enumerate() {
            destination[OBS_INVENTORY + index] = *value as f32 / NORMALIZE_QUANTITY;
        }
        for (index, value) in self.animal_inventory[player].iter().enumerate() {
            destination[OBS_ANIMAL_INVENTORY + index] = *value as f32 / NORMALIZE_QUANTITY;
        }
        for hand in 0..MAX_HANDS {
            let offset = OBS_HAND_INVENTORY + hand * 12;
            for item in 0..PRODUCTS {
                destination[offset + item] =
                    self.hand_inventory[player][hand][item] as f32 / NORMALIZE_QUANTITY;
            }
            for animal in 0..3 {
                destination[offset + 9 + animal] =
                    self.hand_animal_inventory[player][hand][animal] as f32 / NORMALIZE_QUANTITY;
            }
        }
        for farm in 0..PLAYERS {
            let hand_offset = OBS_HAND_POSITIONS + farm * (MAX_HANDS + 1);
            destination[hand_offset] = self.hand_count[farm] as f32 / MAX_HANDS as f32;
            for hand in 0..MAX_HANDS {
                if hand < self.hand_count[farm] {
                    let [x, y] = self.hand_positions[farm][hand];
                    destination[hand_offset + 1 + hand] = (1 + x + 10 * y) as f32 / 100.0;
                }
            }
        }
    }

    fn write_private_observation(&self, player: usize, destination: &mut [f32]) {
        for (index, value) in self.shed[player].iter().enumerate() {
            destination[OBS_SHED + index] = *value as f32 / NORMALIZE_QUANTITY;
        }
        for (index, value) in self.seeds[player].iter().enumerate() {
            destination[OBS_SEEDS + index] = *value as f32 / NORMALIZE_QUANTITY;
        }
        for (index, value) in self.inventory[player].iter().enumerate() {
            destination[OBS_INVENTORY + index] = *value as f32 / NORMALIZE_QUANTITY;
        }
        for (index, value) in self.animal_inventory[player].iter().enumerate() {
            destination[OBS_ANIMAL_INVENTORY + index] = *value as f32 / NORMALIZE_QUANTITY;
        }
        for hand in 0..MAX_HANDS {
            let offset = OBS_HAND_INVENTORY + hand * 12;
            for item in 0..PRODUCTS {
                destination[offset + item] =
                    self.hand_inventory[player][hand][item] as f32 / NORMALIZE_QUANTITY;
            }
            for animal in 0..ANIMAL_TYPES {
                destination[offset + 9 + animal] =
                    self.hand_animal_inventory[player][hand][animal] as f32 / NORMALIZE_QUANTITY;
            }
        }
    }

    fn observe_pair(&self, first: &mut [f32], second: &mut [f32]) {
        self.observe(0, first);
        second.copy_from_slice(first);
        second[3] = 1.0;
        self.write_private_observation(1, second);
    }

    fn write_action_masks(&self, player: usize, destination: &mut [u8]) {
        destination.fill(0);
        let target_none = 0;
        let active_count = self.hand_count[player] + 1;
        for slot in 0..=MAX_HANDS {
            let offset = slot * UNIT_MASK_WIDTH;
            if slot >= active_count {
                destination[offset] = 1;
                destination[offset + UNIT_OPERATIONS.len() + target_none] = 1;
                destination[offset + UNIT_OPERATIONS.len() + 17] = 1;
                continue;
            }
            for operation in 0..UNIT_OPERATIONS.len() {
                destination[offset + operation] = 1;
            }
            let [x, y] = if slot == 0 {
                self.positions[player]
            } else {
                self.hand_positions[player][slot - 1]
            };
            if y <= 0 {
                destination[offset + 1] = 0;
            }
            if y >= BOARD_SIZE as i32 - 1 {
                destination[offset + 2] = 0;
            }
            if x >= BOARD_SIZE as i32 - 1 {
                destination[offset + 3] = 0;
            }
            if x <= 0 {
                destination[offset + 4] = 0;
            }
            let target_offset = offset + UNIT_OPERATIONS.len();
            destination[target_offset + target_none] = 1;
            for item in 0..SHED_ITEMS {
                if self.shed[player][item] > 0 {
                    destination[target_offset + 1 + item] = 1;
                }
            }
            if slot == 0 {
                for item in 0..PRODUCTS {
                    if self.inventory[player][item] > 0 {
                        destination[target_offset + 1 + item] = 1;
                    }
                }
                for animal in 0..ANIMAL_TYPES {
                    if self.animal_inventory[player][animal] > 0 {
                        destination[target_offset + 1 + PRODUCTS + animal] = 1;
                    }
                }
            } else {
                let hand = slot - 1;
                for item in 0..PRODUCTS {
                    if self.hand_inventory[player][hand][item] > 0 {
                        destination[target_offset + 1 + item] = 1;
                    }
                }
                for animal in 0..ANIMAL_TYPES {
                    if self.hand_animal_inventory[player][hand][animal] > 0 {
                        destination[target_offset + 1 + PRODUCTS + animal] = 1;
                    }
                }
            }
            for crop in 0..CROP_TYPES {
                if self.seeds[player][crop] > 0 {
                    destination[target_offset + 1 + crop] = 1;
                }
            }
            let quantity_offset = target_offset + 17;
            destination[quantity_offset..quantity_offset + MAX_QUANTITY as usize + 1].fill(1);
        }

        let market_offset = (MAX_HANDS + 1) * UNIT_MASK_WIDTH;
        let money = self.money[player];
        let shed_total = self.shed_used[player];
        for slot in 0..MAX_MARKET_ORDERS {
            let offset = market_offset + slot * MARKET_MASK_WIDTH;
            for operation in 0..MARKET_OPERATIONS.len() {
                destination[offset + operation] = 1;
            }
            let target_offset = offset + MARKET_OPERATIONS.len();
            destination[target_offset] = 1;
            for crop in 0..CROP_TYPES {
                if money >= CROP_SEED_COSTS[crop] as f32 {
                    destination[target_offset + 1 + crop] = 1;
                }
            }
            if shed_total < self.shed_capacity as i32 {
                for item in [0_usize, 8_usize] {
                    if money >= self.market_prices[item] as f32 {
                        destination[target_offset + 1 + item] = 1;
                    }
                }
                for animal in 0..ANIMAL_TYPES {
                    if money >= ANIMAL_COSTS[animal] as f32 {
                        destination[target_offset + 1 + PRODUCTS + animal] = 1;
                    }
                }
            }
            for item in 0..PRODUCTS {
                if self.shed[player][item] > 0 {
                    destination[target_offset + 1 + item] = 1;
                }
            }
            let quantity_offset = target_offset + 17;
            destination[quantity_offset..quantity_offset + MAX_QUANTITY as usize + 1].fill(1);
            if self.hand_count[player] >= MAX_HANDS
                || money + MONEY_EPSILON < fibonacci(self.hires_today[player]) as f32
            {
                destination[offset + 5] = 0;
            }
            let unlocked_count = self.unlocked[player].iter().filter(|value| **value).count();
            let extra_land = unlocked_count.saturating_sub(1);
            let land_price = LAND_PRICES[extra_land.min(LAND_PRICES.len() - 1)];
            if extra_land >= LAND_PRICES.len() || money < land_price as f32 {
                destination[offset + 6] = 0;
            }
        }
    }
}

const MONEY_EPSILON: f32 = 1.0e-5;

#[pyclass]
struct RustBatchEnv {
    states: Vec<GameState>,
    config: GameConfig,
    // Snapshots captured from live states via `capture_states`, sampled back
    // in via `restore_states` to seed environments from reachable mid-game
    // states instead of only from a fresh `GameState::with_config` reset.
    bank: Vec<GameState>,
}

impl RustBatchEnv {
    fn game_config(&self) -> GameConfig {
        self.config.clone()
    }

    fn observe_all(&self, observation_data: &mut [f32]) {
        const PARALLEL_MIN_ENVS: usize = 128;
        if self.states.len() < PARALLEL_MIN_ENVS {
            for (environment, state) in self.states.iter().enumerate() {
                let start = environment * PLAYERS * OBSERVATION_SIZE;
                let (first, rest) = observation_data[start..].split_at_mut(OBSERVATION_SIZE);
                state.observe_pair(first, &mut rest[..OBSERVATION_SIZE]);
            }
            return;
        }

        self.states
            .par_iter()
            .zip(observation_data.par_chunks_mut(PLAYERS * OBSERVATION_SIZE))
            .for_each(|(state, output)| {
                let (first, rest) = output.split_at_mut(OBSERVATION_SIZE);
                state.observe_pair(first, &mut rest[..OBSERVATION_SIZE]);
            });
    }

    fn masks_all(&self, mask_data: &mut [u8]) {
        const PARALLEL_MIN_ENVS: usize = 128;
        if self.states.len() < PARALLEL_MIN_ENVS {
            for (environment, state) in self.states.iter().enumerate() {
                for player in 0..PLAYERS {
                    let start = (environment * PLAYERS + player) * MASK_SIZE;
                    state.write_action_masks(player, &mut mask_data[start..start + MASK_SIZE]);
                }
            }
            return;
        }

        self.states
            .par_iter()
            .zip(mask_data.par_chunks_mut(PLAYERS * MASK_SIZE))
            .for_each(|(state, output)| {
                for player in 0..PLAYERS {
                    let start = player * MASK_SIZE;
                    state.write_action_masks(player, &mut output[start..start + MASK_SIZE]);
                }
            });
    }

    fn advance(&mut self, actions: &[i64]) {
        const PARALLEL_MIN_ENVS: usize = 128;
        if self.states.len() < PARALLEL_MIN_ENVS {
            for (environment, state) in self.states.iter_mut().enumerate() {
                Self::advance_state(state, environment, actions);
            }
            return;
        }

        self.states
            .par_iter_mut()
            .enumerate()
            .for_each(|(environment, state)| {
                Self::advance_state(state, environment, actions);
            });
    }

    fn advance_state(state: &mut GameState, environment: usize, actions: &[i64]) {
        if state.done {
            return;
        }
        for player in 0..PLAYERS {
            let mut plant_requests = [0usize; 5];
            // Count PLANT requests across every submitted unit slot (farmer +
            // all MAX_HANDS hand slots), not just currently-hired hands.
            // Python's atomic PLANT validation (`_process_actions`'s
            // `plant_demand`/`blocked` in kaggriculture.py) sums PLANT
            // requests over the raw submitted `[farmer_action, *hands_actions]`
            // list before checking which hands actually exist -- a PLANT
            // request from a hand slot beyond the player's hired hand count
            // is a no-op (its position is None) but still counts toward the
            // atomic block for every other PLANT request of that crop this
            // turn. Restricting this count to `0..=state.hand_count[player]`
            // undercounts demand whenever unhired hand slots carry PLANT
            // requests, letting a PLANT through that Python would have
            // blocked (see docs/rl/RUST_ENGINE_PARITY_FINDINGS.md, finding 5).
            for slot in 0..=MAX_HANDS {
                let operation = action_value(actions, environment, player, slot, 0);
                let target = action_value(actions, environment, player, slot, 1);
                if operation == 5 && (0..5).contains(&target) {
                    plant_requests[target as usize] += 1;
                }
            }
            // The atomic block compares demand against the seed count as it
            // stood before ANY of this turn's PLANT actions applied -- not
            // the live, mutating `state.seeds`. Python computes `blocked`
            // once, up front, from the pre-turn `seeds` snapshot, and every
            // unit's PLANT either fully participates or is fully blocked
            // under that one snapshot. Reading `state.seeds[player]` live
            // (as this used to for both the farmer's own check and every
            // hand's check) let the farmer's own successful PLANT decrement
            // the count the very next hand's check compares against, so a
            // demand-equals-supply turn (e.g. 2 requests, 2 seeds) silently
            // blocked every request after the first, even though the whole
            // group should have been allowed.
            let seeds_before_plant = state.seeds[player];
            let operation = action_value(actions, environment, player, 0, 0);
            let target = action_value(actions, environment, player, 0, 1);
            let quantity = action_value(actions, environment, player, 0, 2);
            if operation != 5
                || !(0..5).contains(&target)
                || plant_requests[target as usize] <= seeds_before_plant[target as usize] as usize
            {
                let mut unit = ActiveUnit {
                    position: state.positions[player],
                    inventory: state.inventory[player],
                    animal_inventory: state.animal_inventory[player],
                    item_order: state.farmer_item_order[player],
                };
                state.apply_unit_action(player, &mut unit, operation, target, quantity);
                state.positions[player] = unit.position;
                state.inventory[player] = unit.inventory;
                state.animal_inventory[player] = unit.animal_inventory;
                state.farmer_item_order[player] = unit.item_order;
            }
            for hand in 0..state.hand_count[player] {
                let unit = action_value(actions, environment, player, hand + 1, 0);
                let target = action_value(actions, environment, player, hand + 1, 1);
                if unit == 5
                    && (0..5).contains(&target)
                    && plant_requests[target as usize]
                        > seeds_before_plant[target as usize] as usize
                {
                    continue;
                }
                state.apply_hand_action(
                    player,
                    hand,
                    unit,
                    target,
                    action_value(actions, environment, player, hand + 1, 2),
                );
            }
        }
        for slot in MARKET_ACTION_START..(MARKET_ACTION_START + state.max_market_orders) {
            let operations = [
                action_value(actions, environment, 0, slot, 0),
                action_value(actions, environment, 1, slot, 0),
            ];
            let targets = [
                action_value(actions, environment, 0, slot, 1),
                action_value(actions, environment, 1, slot, 1),
            ];
            for player in 0..PLAYERS {
                let operation = operations[player];
                if operation == 5 || operation == 6 {
                    state.apply_market_action(player, operation, targets[player], 1, None);
                }
            }
            let quantities = [
                action_value(actions, environment, 0, slot, 2).max(0),
                action_value(actions, environment, 1, slot, 2).max(0),
            ];
            let max_quantity = quantities[0].max(quantities[1]);
            let mut active = [
                quantities[0] > 0 && !matches!(operations[0], 5 | 6),
                quantities[1] > 0 && !matches!(operations[1], 5 | 6),
            ];
            let bulkable = |player: usize| {
                (operations[player] == 2 && (targets[player] == 0 || targets[player] == 8))
                    || (operations[player] == 4 && (0..9).contains(&targets[player]))
            };
            if active[0] && active[1] && targets[0] != targets[1] && bulkable(0) && bulkable(1) {
                state.apply_market_bulk(0, operations[0], targets[0], quantities[0]);
                state.apply_market_bulk(1, operations[1], targets[1], quantities[1]);
                continue;
            }
            // Official per-slot lockstep escape: the interpreter aborts the
            // unit loop after 100k iterations (`idx_esc` guard in
            // `_process_market`, kaggriculture.py). Unbounded order
            // quantities make this reachable in principle; mirror it exactly.
            let mut idx_esc: u64 = 0;
            for unit_index in 0..max_quantity {
                idx_esc += 1;
                if idx_esc >= 100_000 {
                    break;
                }
                if !active[0] && !active[1] {
                    break;
                }
                for player in 0..PLAYERS {
                    if unit_index >= quantities[player] {
                        active[player] = false;
                    }
                }
                if active[0] != active[1] && matches!(operations[usize::from(active[1])], 2 | 4) {
                    let player = usize::from(active[1]);
                    state.apply_market_bulk(
                        player,
                        operations[player],
                        targets[player],
                        quantities[player] - unit_index,
                    );
                    break;
                }
                let mut quoted_prices = [None; PLAYERS];
                for player in 0..PLAYERS {
                    if !active[player]
                        || unit_index >= quantities[player]
                        || matches!(operations[player], 5 | 6)
                    {
                        continue;
                    }
                    let operation = operations[player];
                    let target = targets[player];
                    quoted_prices[player] = match operation {
                        2 if (0..9).contains(&target) => Some(state.market_price_integer(
                            target as usize,
                            state.market_inventory[target as usize] as i32 - 1,
                        )),
                        4 if (0..9).contains(&target) => Some(state.market_price_integer(
                            target as usize,
                            state.market_inventory[target as usize] as i32,
                        )),
                        _ => None,
                    };
                }
                for player in 0..PLAYERS {
                    if !active[player]
                        || unit_index >= quantities[player]
                        || matches!(operations[player], 5 | 6)
                    {
                        continue;
                    }
                    if !state.market_action_possible(
                        player,
                        operations[player],
                        targets[player],
                        quoted_prices[player],
                    ) {
                        active[player] = false;
                        continue;
                    }
                    state.apply_market_action(
                        player,
                        operations[player],
                        targets[player],
                        1,
                        quoted_prices[player],
                    );
                }
            }
        }
        state.apply_town_center();
        state.apply_town_shops();
        state.refresh_prices();
        state.decay_plants();
        state.advance_crops_and_animals();
        state.step += 1;
        if state.step + 1 >= state.episode_steps {
            state.done = true;
        }
        #[cfg(debug_assertions)]
        state.debug_assert_invariants();
    }
}

#[pymethods]
impl RustBatchEnv {
    #[new]
    #[pyo3(signature = (
        num_envs,
        episode_steps=720,
        turns_per_day=24,
        weed_spawn_chance=0.005,
        center_interval=24,
        shop_sell_interval=4,
        shop_unlock_interval_days=3,
        starting_money=3000.0,
        max_market_orders=10,
        shed_capacity=100,
        market_params="{}",
        farm_hand_cost_mult=1,
        resolved_config_json=""
    ))]
    // Keep these arguments for compatibility with the Python bridge.
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        episode_steps: u32,
        turns_per_day: u32,
        weed_spawn_chance: f32,
        center_interval: u32,
        shop_sell_interval: u32,
        shop_unlock_interval_days: u32,
        starting_money: f32,
        max_market_orders: usize,
        shed_capacity: usize,
        market_params: &str,
        farm_hand_cost_mult: i32,
        resolved_config_json: &str,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be positive"));
        }
        let mut episode_steps = episode_steps;
        let mut turns_per_day = turns_per_day;
        let mut weed_spawn_chance = weed_spawn_chance;
        let mut center_interval = center_interval;
        let mut shop_sell_interval = shop_sell_interval;
        let mut shop_unlock_interval_days = shop_unlock_interval_days;
        let mut starting_money = starting_money;
        let mut max_market_orders = max_market_orders;
        let mut shed_capacity = shed_capacity;
        let mut farm_hand_cost_mult = farm_hand_cost_mult;
        let mut market_params_text = market_params.to_owned();
        if !resolved_config_json.is_empty() {
            let payload: serde_json::Value =
                serde_json::from_str(resolved_config_json).map_err(|error| {
                    PyValueError::new_err(format!("invalid resolved config: {error}"))
                })?;
            if payload
                .get("schema_version")
                .and_then(serde_json::Value::as_str)
                != Some("kaggriculture-rust-config-v1")
            {
                return Err(PyValueError::new_err("unsupported resolved config schema"));
            }
            let provenance = payload
                .get("provenance")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| PyValueError::new_err("resolved config must contain provenance"))?;
            for field in [
                "engine_version",
                "schema_sha256",
                "source_sha256",
                "rule_tables_sha256",
            ] {
                let value = provenance.get(field).and_then(serde_json::Value::as_str);
                if value.is_none() || value == Some("") {
                    return Err(PyValueError::new_err(format!(
                        "resolved config provenance.{field} must be non-empty"
                    )));
                }
            }
            for (field, expected) in [
                ("engine_version", PINNED_ENGINE_VERSION),
                ("schema_sha256", PINNED_SCHEMA_SHA256),
                ("source_sha256", PINNED_SOURCE_SHA256),
                ("rule_tables_sha256", PINNED_RULE_TABLES_SHA256),
            ] {
                if provenance.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
                    return Err(PyValueError::new_err(format!(
                        "resolved config provenance.{field} does not match the compiled Rust tables"
                    )));
                }
            }
            validate_rule_tables(&payload)?;
            let config = payload
                .get("configuration")
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| {
                    PyValueError::new_err("resolved config must contain configuration")
                })?;
            if config_u32(config, "boardSize")? != BOARD_SIZE as u32 {
                return Err(PyValueError::new_err(format!(
                    "Rust backend supports boardSize={} only",
                    BOARD_SIZE
                )));
            }
            episode_steps = config_u32(config, "episodeSteps")?;
            turns_per_day = config_u32(config, "turnsPerDay")?;
            weed_spawn_chance = config_f32(config, "weedSpawnChance")?;
            center_interval = config_u32(config, "townCenterSellInterval")?;
            shop_sell_interval = config_u32(config, "townShopSellInterval")?;
            shop_unlock_interval_days = config_u32(config, "townShopUnlockInterval")?;
            starting_money = config_f32(config, "startingMoney")?;
            max_market_orders = config_usize(config, "maxMarketOrdersPerTurn")?;
            shed_capacity = config_usize(config, "shedCapacity")?;
            farm_hand_cost_mult = i32::try_from(config_u32(config, "farmHandCostMult")?)
                .map_err(|_| PyValueError::new_err("farmHandCostMult is too large"))?;
            market_params_text = serde_json::to_string(
                config
                    .get("marketParams")
                    .unwrap_or(&serde_json::Value::Object(Default::default())),
            )
            .map_err(|error| PyValueError::new_err(format!("invalid marketParams: {error}")))?;
        }
        if turns_per_day == 0 {
            return Err(PyValueError::new_err("turns_per_day must be positive"));
        }
        if episode_steps == 0 {
            return Err(PyValueError::new_err("episode_steps must be positive"));
        }
        if !starting_money.is_finite() || starting_money < 0.0 {
            return Err(PyValueError::new_err(
                "starting_money must be finite and non-negative",
            ));
        }
        if !(0.0..=1.0).contains(&weed_spawn_chance) {
            return Err(PyValueError::new_err(
                "weed_spawn_chance must be between 0 and 1",
            ));
        }
        if center_interval == 0 || shop_sell_interval == 0 || shop_unlock_interval_days == 0 {
            return Err(PyValueError::new_err("town intervals must be positive"));
        }
        if max_market_orders == 0 || max_market_orders > ACTION_SLOTS - MARKET_ACTION_START {
            return Err(PyValueError::new_err(
                "max_market_orders must be between 1 and 10",
            ));
        }
        if shed_capacity == 0 {
            return Err(PyValueError::new_err("shed_capacity must be positive"));
        }
        if farm_hand_cost_mult < 0 {
            return Err(PyValueError::new_err(
                "farm_hand_cost_mult must be non-negative",
            ));
        }
        let market_config = MarketConfig::from_json(&market_params_text)?;
        let market_tables = Arc::new(MarketLookupTables::build(&market_config));
        let game_config = GameConfig {
            episode_steps,
            turns_per_day,
            weed_spawn_chance,
            center_interval,
            shop_sell_interval,
            shop_unlock_interval_days,
            starting_money,
            max_market_orders,
            shed_capacity,
            farm_hand_cost_mult,
            market_config: market_config.clone(),
            market_tables,
        };
        let states: Vec<GameState> = (0..num_envs)
            .map(|seed| GameState::with_config(seed as u64, game_config.clone()))
            .collect();
        Ok(Self {
            states,
            config: game_config,
            bank: Vec::new(),
        })
    }

    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<ResetOutput<'py>> {
        let seeds = seeds.as_slice()?;
        if seeds.len() != self.states.len() {
            return Err(PyValueError::new_err("seed count must equal num_envs"));
        }
        let game_config = self.game_config();
        self.states = seeds
            .iter()
            .map(|seed| GameState::with_config(*seed, game_config.clone()))
            .collect();
        self.outputs(py)
    }

    #[pyo3(signature=(indices, seeds))]
    fn reset_at<'py>(
        &mut self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'py, i64>,
        seeds: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<ResetOutput<'py>> {
        let indices = indices.as_slice()?;
        let seeds = seeds.as_slice()?;
        if indices.len() != seeds.len() {
            return Err(PyValueError::new_err(
                "indices and seeds must have the same length",
            ));
        }

        let mut seen = HashSet::with_capacity(indices.len());
        let mut validated_indices = Vec::with_capacity(indices.len());
        for &index in indices {
            let index = usize::try_from(index)
                .map_err(|_| PyValueError::new_err("reset indices must be non-negative"))?;
            if index >= self.states.len() {
                return Err(PyValueError::new_err(format!(
                    "reset index {index} is out of range for {} environments",
                    self.states.len()
                )));
            }
            if !seen.insert(index) {
                return Err(PyValueError::new_err(
                    "reset indices must be unique within one call",
                ));
            }
            validated_indices.push(index);
        }

        let game_config = self.game_config();
        for (&index, &seed) in validated_indices.iter().zip(seeds) {
            self.states[index] = GameState::with_config(seed, game_config.clone());
        }
        self.outputs_for_indices(py, &validated_indices)
    }

    /// Clone the given live environments into the state bank and return one
    /// bank slot id per index, in the same order as `indices`.
    fn capture_states<'py>(&mut self, indices: PyReadonlyArray1<'py, i64>) -> PyResult<Vec<u64>> {
        let indices = indices.as_slice()?;
        let mut bank_ids = Vec::with_capacity(indices.len());
        for &index in indices {
            let index = usize::try_from(index)
                .map_err(|_| PyValueError::new_err("capture indices must be non-negative"))?;
            if index >= self.states.len() {
                return Err(PyValueError::new_err(format!(
                    "capture index {index} is out of range for {} environments",
                    self.states.len()
                )));
            }
            self.bank.push(self.states[index].clone());
            bank_ids.push((self.bank.len() - 1) as u64);
        }
        Ok(bank_ids)
    }

    /// Overwrite the given live environments with bank snapshots.
    #[pyo3(signature=(indices, bank_ids))]
    fn restore_states<'py>(
        &mut self,
        py: Python<'py>,
        indices: PyReadonlyArray1<'py, i64>,
        bank_ids: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<ResetOutput<'py>> {
        let indices = indices.as_slice()?;
        let bank_ids = bank_ids.as_slice()?;
        if indices.len() != bank_ids.len() {
            return Err(PyValueError::new_err(
                "indices and bank_ids must have the same length",
            ));
        }

        let mut seen = HashSet::with_capacity(indices.len());
        let mut validated_indices = Vec::with_capacity(indices.len());
        for &index in indices {
            let index = usize::try_from(index)
                .map_err(|_| PyValueError::new_err("restore indices must be non-negative"))?;
            if index >= self.states.len() {
                return Err(PyValueError::new_err(format!(
                    "restore index {index} is out of range for {} environments",
                    self.states.len()
                )));
            }
            if !seen.insert(index) {
                return Err(PyValueError::new_err(
                    "restore indices must be unique within one call",
                ));
            }
            validated_indices.push(index);
        }
        for &bank_id in bank_ids {
            if bank_id as usize >= self.bank.len() {
                return Err(PyValueError::new_err(format!(
                    "bank id {bank_id} is out of range for {} banked states",
                    self.bank.len()
                )));
            }
        }

        for (&index, &bank_id) in validated_indices.iter().zip(bank_ids) {
            self.states[index] = self.bank[bank_id as usize].clone();
        }
        self.outputs_for_indices(py, &validated_indices)
    }

    /// Number of snapshots currently held in the state bank.
    fn bank_size(&self) -> usize {
        self.bank.len()
    }

    /// Discard every banked snapshot. Live environments are unaffected.
    fn clear_bank(&mut self) {
        self.bank.clear();
    }

    /// The in-engine turn counter for every live environment, in order.
    fn steps(&self) -> Vec<u32> {
        self.states.iter().map(|state| state.step).collect()
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray4<'py, i64>,
    ) -> PyResult<StepOutput<'py>> {
        let shape = actions.shape();
        if shape != [self.states.len(), PLAYERS, ACTION_SLOTS, 3] {
            return Err(PyValueError::new_err(format!(
                "actions must have shape ({}, 2, {}, 3)",
                self.states.len(),
                ACTION_SLOTS
            )));
        }
        let actions = actions.as_array();
        let action_data = if let Some(slice) = actions.as_slice() {
            Cow::Borrowed(slice)
        } else {
            Cow::Owned(actions.iter().copied().collect())
        };
        let actions = action_data.as_ref();
        self.advance(actions);
        let (observations, statuses) = self.outputs(py)?;
        let mut rewards = Array2::<f32>::zeros((self.states.len(), PLAYERS));
        for (environment, state) in self.states.iter().enumerate() {
            if state.done {
                rewards[[environment, 0]] = state.money[0];
                rewards[[environment, 1]] = state.money[1];
            }
        }
        let rewards = PyArray2::from_owned_array(py, rewards);
        Ok((observations, rewards, statuses))
    }

    fn step_transition<'py>(&mut self, actions: PyReadonlyArray4<'py, i64>) -> PyResult<()> {
        if actions.shape() != [self.states.len(), PLAYERS, ACTION_SLOTS, ACTION_FIELDS] {
            return Err(PyValueError::new_err(format!(
                "actions must have shape ({}, 2, {}, 3)",
                self.states.len(),
                ACTION_SLOTS
            )));
        }
        let actions = actions.as_array();
        let action_data = if let Some(slice) = actions.as_slice() {
            Cow::Borrowed(slice)
        } else {
            Cow::Owned(actions.iter().copied().collect())
        };
        self.advance(action_data.as_ref());
        Ok(())
    }

    fn observe_into<'py>(&self, mut observations: PyReadwriteArray3<'py, f32>) -> PyResult<()> {
        if observations.shape() != [self.states.len(), PLAYERS, OBSERVATION_SIZE] {
            return Err(PyValueError::new_err(format!(
                "observations must have shape ({}, 2, {})",
                self.states.len(),
                OBSERVATION_SIZE
            )));
        }
        let observation_data = observations
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("observations must be contiguous"))?;
        self.observe_all(observation_data);
        Ok(())
    }

    fn action_masks_into<'py>(&self, mut masks: PyReadwriteArray3<'py, u8>) -> PyResult<()> {
        if masks.shape() != [self.states.len(), PLAYERS, MASK_SIZE] {
            return Err(PyValueError::new_err(format!(
                "masks must have shape ({}, 2, {})",
                self.states.len(),
                MASK_SIZE
            )));
        }
        let mask_data = masks
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("masks must be contiguous"))?;
        self.masks_all(mask_data);
        Ok(())
    }

    #[pyo3(signature=(actions, observations, rewards, statuses))]
    fn step_into<'py>(
        &mut self,
        actions: PyReadonlyArray4<'py, i64>,
        mut observations: PyReadwriteArray3<'py, f32>,
        mut rewards: PyReadwriteArray2<'py, f32>,
        mut statuses: PyReadwriteArray2<'py, u8>,
    ) -> PyResult<()> {
        if actions.shape() != [self.states.len(), PLAYERS, ACTION_SLOTS, ACTION_FIELDS] {
            return Err(PyValueError::new_err(format!(
                "actions must have shape ({}, 2, {}, 3)",
                self.states.len(),
                ACTION_SLOTS
            )));
        }
        if observations.shape() != [self.states.len(), PLAYERS, OBSERVATION_SIZE] {
            return Err(PyValueError::new_err(format!(
                "observations must have shape ({}, 2, {})",
                self.states.len(),
                OBSERVATION_SIZE
            )));
        }
        if rewards.shape() != [self.states.len(), PLAYERS] {
            return Err(PyValueError::new_err(format!(
                "rewards must have shape ({}, 2)",
                self.states.len()
            )));
        }
        if statuses.shape() != [self.states.len(), PLAYERS] {
            return Err(PyValueError::new_err(format!(
                "statuses must have shape ({}, 2)",
                self.states.len()
            )));
        }
        let actions = actions.as_array();
        let action_data = if let Some(slice) = actions.as_slice() {
            Cow::Borrowed(slice)
        } else {
            Cow::Owned(actions.iter().copied().collect())
        };
        self.advance(action_data.as_ref());

        let observation_data = observations
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("observations must be contiguous"))?;
        self.observe_all(observation_data);
        let reward_data = rewards
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("rewards must be contiguous"))?;
        reward_data.fill(0.0);
        let status_data = statuses
            .as_slice_mut()
            .map_err(|_| PyRuntimeError::new_err("statuses must be contiguous"))?;
        status_data.fill(0);
        for (environment, state) in self.states.iter().enumerate() {
            if state.done {
                reward_data[environment * PLAYERS] = state.money[0];
                reward_data[environment * PLAYERS + 1] = state.money[1];
                status_data[environment * PLAYERS] = 1;
                status_data[environment * PLAYERS + 1] = 1;
            }
        }
        Ok(())
    }

    fn num_envs(&self) -> usize {
        self.states.len()
    }
}

impl RustBatchEnv {
    fn outputs_for_indices<'py>(
        &self,
        py: Python<'py>,
        indices: &[usize],
    ) -> PyResult<ResetOutput<'py>> {
        let mut observations = Array3::<f32>::zeros((indices.len(), PLAYERS, OBSERVATION_SIZE));
        let observation_data = observations
            .as_slice_mut()
            .ok_or_else(|| PyRuntimeError::new_err("observation buffer is not contiguous"))?;
        for (output_index, &environment) in indices.iter().enumerate() {
            let start = output_index * PLAYERS * OBSERVATION_SIZE;
            let (first, rest) = observation_data[start..].split_at_mut(OBSERVATION_SIZE);
            self.states[environment].observe_pair(first, &mut rest[..OBSERVATION_SIZE]);
        }
        let observation_array = PyArray3::from_owned_array(py, observations);
        let mut statuses = Array2::<u8>::zeros((indices.len(), PLAYERS));
        for (output_index, &environment) in indices.iter().enumerate() {
            if self.states[environment].done {
                statuses[[output_index, 0]] = 1;
                statuses[[output_index, 1]] = 1;
            }
        }
        let statuses = PyArray2::from_owned_array(py, statuses);
        Ok((observation_array, statuses))
    }

    fn outputs<'py>(&self, py: Python<'py>) -> PyResult<ResetOutput<'py>> {
        let mut observations = Array3::<f32>::zeros((self.states.len(), PLAYERS, OBSERVATION_SIZE));
        let observation_data = observations
            .as_slice_mut()
            .ok_or_else(|| PyRuntimeError::new_err("observation buffer is not contiguous"))?;
        self.observe_all(observation_data);
        let observation_array = PyArray3::from_owned_array(py, observations);
        let mut statuses = Array2::<u8>::zeros((self.states.len(), PLAYERS));
        for (environment, state) in self.states.iter().enumerate() {
            if state.done {
                statuses[[environment, 0]] = 1;
                statuses[[environment, 1]] = 1;
            }
        }
        let statuses = PyArray2::from_owned_array(py, statuses);
        Ok((observation_array, statuses))
    }
}

#[pymodule]
fn _kaggriculture_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBatchEnv>()?;
    m.add("OBS_SIZE", OBSERVATION_SIZE)?;
    m.add("MASK_SIZE", MASK_SIZE)?;
    m.add("MAX_HANDS", MAX_HANDS)?;
    m.add("ACTION_SLOTS", ACTION_SLOTS)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_game_config() -> GameConfig {
        GameConfig {
            episode_steps: 720,
            turns_per_day: 24,
            weed_spawn_chance: 0.0,
            center_interval: 24,
            shop_sell_interval: 4,
            shop_unlock_interval_days: 3,
            starting_money: 3000.0,
            max_market_orders: 10,
            shed_capacity: 100,
            farm_hand_cost_mult: 1,
            market_config: MarketConfig::default(),
            market_tables: Arc::new(MarketLookupTables::build(&MarketConfig::default())),
        }
    }

    #[test]
    fn advance_farm_growth_never_spawns_weeds_even_with_high_weed_spawn_chance() {
        let mut config = test_game_config();
        config.weed_spawn_chance = 1.0;
        let mut state = GameState::with_config(0, config);
        // Tile 0 (NW quadrant, unlocked by default) is empty, structure-free,
        // and not already weeded -- eligible for spawn if weed-spawn ran.
        assert!(state.tile_unlocked(0, 0));
        state.advance_farm_growth(0);
        assert!(
            !state.weed[0][0],
            "advance_farm_growth must never spawn weeds itself -- that stays in the \
             caller so coast-projection can call this method without it"
        );
    }

    #[test]
    fn production_advance_crops_and_animals_still_spawns_weeds_after_the_split() {
        // Regression guard for the growth/weed-spawn split above: the
        // production entry point must still spawn weeds exactly as
        // before, proving the split didn't silently drop that behavior
        // from the real per-turn path.
        let mut config = test_game_config();
        config.weed_spawn_chance = 1.0;
        let mut state = GameState::with_config(0, config);
        state.step = state.turns_per_day - 1; // end_of_day on this call
        assert!(state.tile_unlocked(0, 0));
        state.advance_crops_and_animals();
        assert!(
            state.weed[0][0],
            "production advance_crops_and_animals must still spawn weeds at a day boundary"
        );
    }

    #[test]
    fn harvest_drains_all_accumulated_animal_yield_in_one_call() {
        // Regression test: HARVEST on an animal tile used to take only 1
        // unit of accumulated yield per call (self.animal_yield -= 1),
        // unlike the crop branch a few lines below it in the same match
        // arm, which correctly drains the whole accumulated yield at
        // once. The official Python engine's animal HARVEST does the
        // same (`units = tile["yield_units"]; tile["yield_units"] = 0`).
        // A player who lets yield accumulate across several days (the
        // normal way to play, since HARVEST costs a turn) would recover
        // only a fraction of what the real game gives them.
        let mut state = GameState::with_config(0, test_game_config());
        let tile = tile_index(state.positions[0]);
        state.animal_on_tile[0][tile] = true;
        state.animal_kind[0][tile] = 2; // SHEEP
        state.animal_yield[0][tile] = 3;
        let mut unit = ActiveUnit {
            position: state.positions[0],
            inventory: state.inventory[0],
            animal_inventory: state.animal_inventory[0],
            item_order: [0; SHED_ITEMS],
        };
        state.apply_unit_action(0, &mut unit, 6, 0, 0);
        state.inventory[0] = unit.inventory;
        assert_eq!(
            state.animal_yield[0][tile], 0,
            "one HARVEST must drain all accumulated yield, not just 1 unit"
        );
        assert_eq!(
            state.inventory[0][ANIMAL_PRODUCT[2]], 3,
            "the full accumulated yield must land in carried inventory in one call"
        );
    }

    #[test]
    fn drop_discards_carried_inventory_that_does_not_fit_in_a_full_shed() {
        // Regression test: DROP used to keep whatever didn't fit in the
        // shed in the farmer's hands (`unit.inventory[item] -= moved`),
        // like PICKUP/PLACE-to-shed correctly do for their genuinely
        // room-limited partial transfers. But Python's DROP handler
        // (kaggriculture.py) unconditionally clears the whole carried
        // inventory after computing shed room (`del inv[item]` runs
        // regardless of how much fit), silently destroying the excess
        // instead of keeping it. Rust must match that quirk.
        let mut config = test_game_config();
        config.shed_capacity = 1;
        let mut state = GameState::with_config(0, config);
        assert!(state.is_shed_adjacent_at(state.positions[0]));
        state.shed_used[0] = 1; // shed already full, zero room left
        let mut unit = ActiveUnit {
            position: state.positions[0],
            inventory: state.inventory[0],
            animal_inventory: state.animal_inventory[0],
            item_order: [0; SHED_ITEMS],
        };
        unit.inventory[0] = 5; // carrying 5 units the shed has no room for
        state.apply_unit_action(0, &mut unit, 15, 0, 0);
        assert_eq!(
            unit.inventory[0], 0,
            "DROP must clear the whole carried inventory even when it doesn't fit in the shed"
        );
        assert_eq!(
            state.shed[0][0], 0,
            "the unfit units must not have been moved into the shed"
        );
    }

    #[test]
    fn drop_gives_shed_room_priority_to_whichever_item_was_picked_up_first() {
        // Regression test for finding 11: Python's DROP iterates the
        // farmer's inventory `dict` in INSERTION order, not a fixed
        // products-then-animals order. A unit that PICKUP'd SHEEP before
        // GOOSE must, on a room-limited DROP, give SHEEP priority for the
        // scarce shed room -- even though GOOSE has a lower fixed index
        // (9) than SHEEP (11) in the underlying array representation.
        let mut config = test_game_config();
        config.shed_capacity = 20;
        let mut state = GameState::with_config(0, config);
        assert!(state.is_shed_adjacent_at(state.positions[0]));
        state.shed[0][11] = 3; // SHEEP in the shed, ready to pick up
        state.shed[0][9] = 2; // GOOSE in the shed, ready to pick up
        state.shed_used[0] = 5;
        let mut unit = ActiveUnit {
            position: state.positions[0],
            inventory: state.inventory[0],
            animal_inventory: state.animal_inventory[0],
            item_order: [0; SHED_ITEMS],
        };
        state.apply_unit_action(0, &mut unit, 11, 3, 3); // PICKUP SHEEP x3 first
        state.apply_unit_action(0, &mut unit, 11, 1, 2); // PICKUP GOOSE x2 second
        assert_eq!(
            unit.animal_inventory,
            [2, 0, 3],
            "GOOSE=2, COW=0, SHEEP=3 carried"
        );
        // Shrink shed room to exactly 3 -- enough for SHEEP alone, none
        // left over for GOOSE.
        state.shed_used[0] = state.shed_capacity as i32 - 3;
        state.apply_unit_action(0, &mut unit, 15, 0, 0); // DROP
        assert_eq!(
            unit.animal_inventory,
            [0, 0, 0],
            "DROP must clear all carried animals regardless of what fit"
        );
        assert_eq!(
            state.shed[0][11], 3,
            "SHEEP (picked up first) must get priority for the 3 units of shed room"
        );
        assert_eq!(
            state.shed[0][9], 0,
            "GOOSE (picked up second) must get none of the shed room once SHEEP took it all"
        );
    }

    #[test]
    fn center_consumes_eight_products_once_per_day() {
        let mut state = GameState::with_config(
            0,
            GameConfig {
                episode_steps: 720,
                turns_per_day: 24,
                weed_spawn_chance: 0.005,
                center_interval: 24,
                shop_sell_interval: 4,
                shop_unlock_interval_days: 3,
                starting_money: 3000.0,
                max_market_orders: 10,
                shed_capacity: 100,
                farm_hand_cost_mult: 1,
                market_config: MarketConfig::default(),
                market_tables: Arc::new(MarketLookupTables::build(&MarketConfig::default())),
            },
        );
        state.apply_town_center();
        assert_eq!(state.market_inventory[0], 9_999.0);
        assert_eq!(state.market_inventory[7], 9_999.0);
        assert_eq!(state.market_inventory[8], 10_000.0);
        state.step = 1;
        state.apply_town_center();
        assert_eq!(state.market_inventory[0], 9_999.0);
    }

    #[test]
    fn market_lookup_matches_reference_prices() {
        let config = MarketConfig::default();
        let tables = MarketLookupTables::build(&config);
        for item in 0..PRODUCTS {
            for inventory in [0.0, 1.0, 9_999.0, 10_000.0, 10_001.0, 20_000.0] {
                assert_eq!(
                    tables.get(item, inventory),
                    Some(market_price_reference(&config, item, inventory))
                );
            }
            assert_eq!(
                tables.get(item, 10_000.5),
                None,
                "fractional inventory must use the reference path"
            );
        }
    }

    #[test]
    fn market_bulk_matches_per_unit_reference_for_buy_and_sell() {
        let config = GameConfig {
            episode_steps: 720,
            turns_per_day: 24,
            weed_spawn_chance: 0.005,
            center_interval: 24,
            shop_sell_interval: 4,
            shop_unlock_interval_days: 3,
            starting_money: 100_000.0,
            max_market_orders: 10,
            shed_capacity: 100,
            farm_hand_cost_mult: 1,
            market_config: MarketConfig::default(),
            market_tables: Arc::new(MarketLookupTables::build(&MarketConfig::default())),
        };
        let mut bulk_buy = GameState::with_config(0, config.clone());
        let mut reference_buy = GameState::with_config(0, config.clone());
        bulk_buy.apply_market_bulk(0, 2, 0, 25);
        for _ in 0..25 {
            let price = market_price_reference(
                &reference_buy.market_config,
                0,
                reference_buy.market_inventory[0] - 1.0,
            );
            reference_buy.apply_market_action(0, 2, 0, 1, Some(price));
        }
        assert_eq!(bulk_buy.money, reference_buy.money);
        assert_eq!(bulk_buy.shed, reference_buy.shed);
        assert_eq!(bulk_buy.market_inventory, reference_buy.market_inventory);

        let mut bulk_sell = GameState::with_config(0, config.clone());
        let mut reference_sell = GameState::with_config(0, config);
        bulk_sell.shed[0][0] = 25;
        bulk_sell.shed_used[0] = 25;
        reference_sell.shed[0][0] = 25;
        reference_sell.shed_used[0] = 25;
        bulk_sell.apply_market_bulk(0, 4, 0, 25);
        for _ in 0..25 {
            let price = market_price_reference(
                &reference_sell.market_config,
                0,
                reference_sell.market_inventory[0],
            );
            reference_sell.apply_market_action(0, 4, 0, 1, Some(price));
        }
        assert_eq!(bulk_sell.money, reference_sell.money);
        assert_eq!(bulk_sell.shed, reference_sell.shed);
        assert_eq!(bulk_sell.market_inventory, reference_sell.market_inventory);
    }

    #[test]
    fn two_player_simultaneous_sell_of_same_product_matches_interleaved_reference() {
        // Two players selling the same item in the same turn must be quoted
        // at the same pre-round price and both commit in player order, even
        // once the price crosses the PRICE_FLOOR (where a naive bulk-style
        // shortcut could double-count or skip a supply increment). MILK is
        // PRODUCTS index 6; I0=10000, above_target 1.60, base 160, T 122
        // gives amp ~2.0984/unit, so a 12-unit sale starting 70 above I0
        // straddles the point where price stops advancing market_inventory.
        let config = test_game_config();
        let mut state = GameState::with_config(0, config);
        state.shed[0][6] = 12;
        state.shed_used[0] = 12;
        state.shed[1][6] = 12;
        state.shed_used[1] = 12;
        state.market_inventory[6] = 10_070.0;
        state.refresh_prices();

        let mut actions = vec![0_i64; ACTION_ENV_STRIDE];
        for player in 0..PLAYERS {
            let base = player * ACTION_PLAYER_STRIDE + MARKET_ACTION_START * ACTION_FIELDS;
            actions[base] = 4; // SELL
            actions[base + 1] = 6; // MILK
            actions[base + 2] = 12;
        }
        RustBatchEnv::advance_state(&mut state, 0, &actions);

        // Reference: exact reimplementation of the Python engine's per-unit
        // lockstep loop for two players selling the same item -- both quoted
        // at the same pre-round inventory, then both committed in player
        // order. Town center / shop consumption also touches
        // market_inventory once per turn, so only revenue is compared here.
        let mut ref_money = [3000.0_f32, 3000.0_f32];
        let mut ref_inventory = 10_070.0_f64;
        for _ in 0..12 {
            let price = market_price_reference(&MarketConfig::default(), 6, ref_inventory);
            for player in 0..PLAYERS {
                ref_money[player] += price as f32;
            }
            if price > 1 {
                ref_inventory += 2.0;
            }
        }

        assert_eq!(state.money[0], ref_money[0], "player 0 revenue mismatch");
        assert_eq!(state.money[1], ref_money[1], "player 1 revenue mismatch");
    }

    #[test]
    fn animal_base_yield_accrues_even_when_not_fed_on_production_day() {
        // Regression test for docs/rl/RUST_ENGINE_PARITY_FINDINGS.md finding
        // 4 (misdiagnosed during investigation as a MILK sell-price drift;
        // root cause traced to first divergence at turn 306 of real episode
        // 92478595, where a COW production day landed on an unfed day).
        //
        // The Python engine's `_daily_refresh_animals` always credits the
        // base +1 yield on a production-interval day; `fed_today` only
        // gates the pending-care *bonus*, not the base yield itself:
        //   base = 1
        //   bonus = tile.pop("pending_care_bonus", 0) if tile["fed_today"] else 0
        //   tile["yield_units"] = min(max_held, tile["yield_units"] + base + bonus)
        // The Rust port incorrectly required `animal_fed` for the entire
        // block, silently dropping the base yield on any unfed production
        // day -- a real, if intermittent, production-parity bug distinct
        // from the earlier HARVEST-draining bug.
        let config = test_game_config();
        let mut state = GameState::with_config(0, config);
        let tile = 3usize; // PASTURE-eligible tile in the unlocked NW quadrant.
        state.structure[0][tile] = true;
        state.structure_kind[0][tile] = 1; // PASTURE
        state.animal_on_tile[0][tile] = true;
        state.animal_kind[0][tile] = 1; // COW: first_yield_day 8, interval 2.
        state.animal_age[0][tile] = ANIMAL_FIRST_YIELD_DAY[1] - 1; // becomes first_yield_day after +=1.
        state.animal_fed[0][tile] = false; // Not fed on the production day itself.
        state.animal_cared[0][tile] = false;
        state.animal_yield[0][tile] = 0;
        state.animal_pending_care[0][tile] = 0;

        state.advance_farm_growth(0);

        assert_eq!(
            state.animal_yield[0][tile], 1,
            "base yield must accrue on a production day even when unfed"
        );
    }

    #[test]
    fn plant_allows_every_request_when_demand_exactly_equals_seed_supply() {
        // Regression test for docs/rl/RUST_ENGINE_PARITY_FINDINGS.md finding
        // 9, found by a fresh fuzz seed after findings 5/7/8 were fixed:
        // the atomic PLANT-request gate compared demand against the LIVE,
        // mutating `state.seeds[player]` instead of a snapshot taken before
        // any of this turn's PLANT actions applied. With 2 simultaneous
        // WHEAT PLANT requests (farmer + one hand, at different tiles) and
        // exactly 2 WHEAT seeds available, the farmer's own PLANT (applied
        // first) legitimately consumed the first seed, dropping the live
        // count to 1 -- so the hand's gate check then compared demand (2)
        // against the ALREADY-DECREMENTED count (1), incorrectly blocking
        // a request that should have been allowed under the original,
        // pre-turn seed count. Python computes its `blocked` set once, up
        // front, from the pre-turn seed snapshot, so a demand-equals-supply
        // turn is never partially blocked.
        let mut state = GameState::with_config(0, test_game_config());
        state.hand_count[0] = 1;
        state.seeds[0][0] = 2; // exactly 2 WHEAT seeds: demand == supply.
        let farmer_tile = tile_index(state.positions[0]);
        // A distinct, empty tile within the unlocked NW quadrant (x<5,
        // y<5) -- PLANT requires `tile_unlocked`, so an adjacent tile
        // outside the starting quadrant would silently no-op regardless
        // of the seed-gate logic under test.
        state.hand_positions[0][0] = [3, 4];
        let hand_tile = tile_index(state.hand_positions[0][0]);
        assert_ne!(
            farmer_tile, hand_tile,
            "test fixture requires two distinct tiles"
        );

        let mut actions = vec![0_i64; ACTION_ENV_STRIDE];
        let base = 0 * ACTION_PLAYER_STRIDE; // player 0, farmer slot (0).
        actions[base] = 5; // PLANT
        actions[base + 1] = 0; // WHEAT
        let hand_base = 0 * ACTION_PLAYER_STRIDE + 1 * ACTION_FIELDS; // player 0, hand slot (1).
        actions[hand_base] = 5; // PLANT
        actions[hand_base + 1] = 0; // WHEAT

        RustBatchEnv::advance_state(&mut state, 0, &actions);

        assert_ne!(
            state.crop_age[0][farmer_tile], EMPTY_CROP_AGE,
            "the farmer's own PLANT must succeed when demand equals supply"
        );
        assert_ne!(
            state.crop_age[0][hand_tile], EMPTY_CROP_AGE,
            "the hand's PLANT must also succeed when demand equals supply, \
             not be blocked by the farmer's own seed decrement"
        );
        assert_eq!(state.seeds[0][0], 0, "both seeds must be consumed");
    }

    #[test]
    fn plant_resets_a_stale_fertilized_until_day_from_a_prior_crop() {
        // Regression test for docs/rl/RUST_ENGINE_PARITY_FINDINGS.md finding
        // 8, found by widening the random fuzzer past the checked-in
        // suite's turn counts: PLANT (op 5) reset crop_age, crop_kind,
        // crop_decay_step, crop_yield, crop_unwatered, and weed, but never
        // crop_fertilized_until_day. Python's `_new_plant` always builds a
        // brand-new tile dict with `"fertilized_until_day": -1`, so a fresh
        // planting can never inherit a prior crop's fertilizer status. A
        // tile fertilized under an earlier crop, then harvested, decayed,
        // or dug, left a stale future `crop_fertilized_until_day` in place;
        // replanting there without this fix silently carried that bonus
        // into a crop that was never fertilized.
        let mut state = GameState::with_config(0, test_game_config());
        let tile = tile_index(state.positions[0]);
        // Simulate a tile left over from an earlier, fertilized crop: no
        // crop currently occupies it (as if harvested/decayed/dug), but
        // its fertilized-until-day field is still a future day.
        state.crop_age[0][tile] = EMPTY_CROP_AGE;
        state.crop_kind[0][tile] = EMPTY_KIND;
        state.crop_fertilized_until_day[0][tile] = 5;
        state.seeds[0][0] = 1; // one WHEAT seed available.
        let mut unit = ActiveUnit {
            position: state.positions[0],
            inventory: state.inventory[0],
            animal_inventory: state.animal_inventory[0],
            item_order: [0; SHED_ITEMS],
        };
        state.apply_unit_action(0, &mut unit, 5, 0, 0); // PLANT WHEAT.
        assert_eq!(
            state.crop_fertilized_until_day[0][tile], -1,
            "a freshly planted crop must not inherit a prior crop's fertilizer bonus"
        );
    }

    #[test]
    fn plant_resets_a_stale_watered_today_flag_from_a_prior_crop() {
        // Regression test for finding 12: PLANT (op 5) never reset
        // crop_watered_today, the same class of leak as finding 8's
        // crop_fertilized_until_day bug. A single-harvest crop
        // (CROP_INTERVAL == 0) that gets WATERed, then HARVESTed --
        // HARVEST clears crop_age/crop_kind but not crop_watered_today --
        // left the tile stuck at "already watered today" for whatever
        // gets PLANTed there next this same day. Python's `_new_plant`
        // always builds a fresh tile dict with `"watered_today": False`,
        // so a freshly planted crop can never inherit a prior crop's
        // already-watered status.
        let mut state = GameState::with_config(0, test_game_config());
        let tile = tile_index(state.positions[0]);
        // Simulate a tile left over from an earlier crop that was
        // watered today, then harvested/decayed/dug (no crop currently
        // occupies it, but crop_watered_today is still stuck at true).
        state.crop_age[0][tile] = EMPTY_CROP_AGE;
        state.crop_kind[0][tile] = EMPTY_KIND;
        state.crop_watered_today[0][tile] = true;
        state.seeds[0][0] = 1; // one WHEAT seed available.
        let mut unit = ActiveUnit {
            position: state.positions[0],
            inventory: state.inventory[0],
            animal_inventory: state.animal_inventory[0],
            item_order: [0; SHED_ITEMS],
        };
        state.apply_unit_action(0, &mut unit, 5, 0, 0); // PLANT WHEAT.
        assert!(
            !state.crop_watered_today[0][tile],
            "a freshly planted crop must not inherit a prior crop's already-watered status"
        );
    }

    #[test]
    fn python_random_matches_seeded_reference_values() {
        let mut random = PythonRandom::new(7);
        assert!((random.random() - 0.32383276).abs() < 1e-7);
        assert!((random.random() - 0.15084917).abs() < 1e-7);
        assert!((random.random() - 0.65093446).abs() < 1e-7);
        let mut random = PythonRandom::new(7);
        assert_eq!(random.choice(8), 5);
        assert_eq!(random.choice(8), 2);
        assert_eq!(random.choice(8), 6);
    }

    #[test]
    fn python_random_preserves_large_supported_seed_values() {
        let seed = ((u64::MAX as u128) * 1_000_003) ^ 29;
        let mut random = PythonRandom::new(seed);
        assert!((random.random() - 0.014_301_036).abs() < 1e-8);
        assert!((random.random() - 0.802_787_8).abs() < 1e-8);
        assert!((random.random() - 0.113_463_07).abs() < 1e-8);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn debug_invariants_reject_shed_counter_drift() {
        let mut state = GameState::with_config(
            0,
            GameConfig {
                episode_steps: 720,
                turns_per_day: 24,
                weed_spawn_chance: 0.005,
                center_interval: 24,
                shop_sell_interval: 4,
                shop_unlock_interval_days: 3,
                starting_money: 3000.0,
                max_market_orders: 10,
                shed_capacity: 100,
                farm_hand_cost_mult: 1,
                market_config: MarketConfig::default(),
                market_tables: Arc::new(MarketLookupTables::build(&MarketConfig::default())),
            },
        );
        state.shed_used[0] = 1;
        state.debug_assert_invariants();
    }
}
