"""Generate Rust protocol and rule constants from the installed Kaggriculture engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "rust/kaggriculture_env/src/generated_protocol.rs"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rust_string(value: str) -> str:
    return json.dumps(value)


def _rust_array(values: list[Any], formatter=str) -> str:
    return "[" + ", ".join(formatter(value) for value in values) + "]"


def _default(configuration: dict[str, Any], name: str) -> Any:
    value = configuration[name]
    return value.get("default") if isinstance(value, dict) else value


def _rust_float(value: float | int) -> str:
    rendered = f"{float(value):.12g}"
    return rendered if any(mark in rendered for mark in ".eE") else rendered + ".0"


def _rust_bool(value: bool) -> str:
    return "true" if value else "false"


def _load_source() -> tuple[Any, dict[str, Any], Path, Path]:
    try:
        from kaggle_environments.envs.kaggriculture import kaggriculture as source
    except ImportError as error:  # pragma: no cover - depends on the active environment
        raise SystemExit(
            "The kaggriculture Conda environment is required. "
            "Run this script with `conda run -n kaggriculture`."
        ) from error
    source_path = Path(source.__file__)
    schema_path = source_path.with_suffix(".json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return source, schema, source_path, schema_path


def _rule_tables(source: Any, schema: dict[str, Any]) -> dict[str, Any]:
    crops = source.CROPS
    animals = source.ANIMALS
    shops = source.SHOPS
    products = list(source.PRODUCTS)
    crop_names = list(crops)
    animal_names = list(animals)
    shop_names = list(shops)
    shop_engine_order = sorted(shop_names)
    unit_operations = [
        "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "DROP",
        "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE",
        "FEED", "COLLECT_FERTILIZER", "CARE", "DIG",
    ]
    market_operations = ["PASS", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"]
    product_index = {name: index for index, name in enumerate(products)}
    animal_product = [product_index[animals[name]["product"]] for name in animal_names]
    animal_structure = [0 if animals[name]["structure"] == "COOP" else 1 for name in animal_names]
    shop_demands = [
        [product_index[item] for item in shops[name]] + [-1] * (4 - len(shops[name]))
        for name in shop_engine_order
    ]
    rule_tables = {
        "products": products,
        "unit_operations": unit_operations,
        "market_operations": market_operations,
        "crops": crops,
        "animals": animals,
        "shops": shops,
        "shop_engine_order": shop_engine_order,
        "land_order": ["NE", "SW", "SE"],
        "land_prices": list(source.LAND_PRICES),
        "town_center_products": list(source.TOWN_CENTER_PRODUCTS),
        "max_shop_instances": source.MAX_SHOP_INSTANCES,
        "town_center_demand_schedule": [[0, 1]],
        "max_hands": 16,
        "max_market_orders": _default(schema["configuration"], "maxMarketOrdersPerTurn"),
        "max_quantity": 100,
        "board_size": _default(schema["configuration"], "boardSize"),
        "market_params": source.MARKET_PARAMS,
    }
    return {
        "tables": rule_tables,
        "crop_names": crop_names,
        "animal_names": animal_names,
        "shop_names": shop_names,
        "products": products,
        "unit_operations": unit_operations,
        "market_operations": market_operations,
        "shop_engine_order": shop_engine_order,
        "animal_product": animal_product,
        "animal_structure": animal_structure,
        "shop_demands": shop_demands,
    }


def render() -> str:
    source, schema, source_path, schema_path = _load_source()
    data = _rule_tables(source, schema)
    tables = data["tables"]
    canonical_tables = json.dumps(tables, sort_keys=True, separators=(",", ":"))
    source_hash = _sha256(source_path)
    schema_hash = _sha256(schema_path)
    rule_hash = hashlib.sha256(canonical_tables.encode("utf-8")).hexdigest()
    config = schema["configuration"]
    market = source.MARKET_PARAMS
    market_order = data["products"]

    lines = [
        "// Generated from the installed Kaggriculture Conda environment. Do not edit by hand.",
        "#![allow(dead_code)]",
        f'pub const ENGINE_VERSION: &str = {_rust_string(str(source.specification["version"]))};',
        f'pub const SOURCE_SHA256: &str = {_rust_string(source_hash)};',
        f'pub const SCHEMA_SHA256: &str = {_rust_string(schema_hash)};',
        f'pub const RULE_TABLES_SHA256: &str = {_rust_string(rule_hash)};',
        f"pub const BOARD_SIZE: usize = {tables['board_size']};",
        f"pub const MAX_HANDS: usize = {tables['max_hands']};",
        f"pub const MAX_SHOP_INSTANCES: usize = {source.MAX_SHOP_INSTANCES};",
        f"pub const MAX_MARKET_ORDERS: usize = {_default(config, 'maxMarketOrdersPerTurn')};",
        f"pub const MAX_QUANTITY: i64 = {tables['max_quantity']};",
        f"pub const SEASON_STEPS: usize = {_default(config, 'episodeSteps')};",
        "pub const SEASON_DAYS: usize = 30;",
        f"pub const TURNS_PER_DAY: usize = {_default(config, 'turnsPerDay')};",
        f"pub const UNIT_OPERATIONS: [&str; {len(data['unit_operations'])}] = {_rust_array(data['unit_operations'], _rust_string)};",
        f"pub const MARKET_OPERATIONS: [&str; {len(data['market_operations'])}] = {_rust_array(data['market_operations'], _rust_string)};",
        f"pub const CROPS: [&str; {len(data['crop_names'])}] = {_rust_array(data['crop_names'], _rust_string)};",
        f"pub const PRODUCTS: [&str; {len(data['products'])}] = {_rust_array(data['products'], _rust_string)};",
        f"pub const ANIMALS: [&str; {len(data['animal_names'])}] = {_rust_array(data['animal_names'], _rust_string)};",
        f"pub const SHOPS: [&str; {len(data['shop_names'])}] = {_rust_array(data['shop_names'], _rust_string)};",
        f"pub const SHOP_ENGINE_ORDER: [&str; {len(data['shop_engine_order'])}] = {_rust_array(data['shop_engine_order'], _rust_string)};",
        f"pub const CROP_SEED_COSTS: [i32; {len(data['crop_names'])}] = {_rust_array([source.CROPS[name]['seed'] for name in data['crop_names']])};",
        f"pub const CROP_FIRST_YIELD_DAY: [i32; {len(data['crop_names'])}] = {_rust_array([source.CROPS[name]['first_yield_day'] for name in data['crop_names']])};",
        f"pub const CROP_MAX_YIELD_DAY: [i32; {len(data['crop_names'])}] = {_rust_array([source.CROPS[name]['max_yield_day'] for name in data['crop_names']])};",
        f"pub const CROP_MAX_YIELD: [i32; {len(data['crop_names'])}] = {_rust_array([source.CROPS[name]['max_yield'] for name in data['crop_names']])};",
        f"pub const CROP_INTERVAL: [i32; {len(data['crop_names'])}] = {_rust_array([source.CROPS[name]['interval'] for name in data['crop_names']])};",
        f"pub const CROP_ONGOING: [bool; {len(data['crop_names'])}] = {_rust_array([source.CROPS[name]['ongoing'] for name in data['crop_names']], _rust_bool)};",
        f"pub const ANIMAL_COSTS: [i32; {len(data['animal_names'])}] = {_rust_array([source.ANIMALS[name]['cost'] for name in data['animal_names']])};",
        f"pub const ANIMAL_STRUCTURE: [i8; {len(data['animal_names'])}] = {_rust_array(data['animal_structure'])};",
        f"pub const ANIMAL_FIRST_YIELD_DAY: [i32; {len(data['animal_names'])}] = {_rust_array([source.ANIMALS[name]['first_yield_day'] for name in data['animal_names']])};",
        f"pub const ANIMAL_INTERVAL: [i32; {len(data['animal_names'])}] = {_rust_array([source.ANIMALS[name]['interval'] for name in data['animal_names']])};",
        f"pub const ANIMAL_MAX_HELD: [i32; {len(data['animal_names'])}] = {_rust_array([source.ANIMALS[name]['max_held'] for name in data['animal_names']])};",
        f"pub const ANIMAL_PRODUCT: [usize; {len(data['animal_names'])}] = {_rust_array(data['animal_product'])};",
        "pub const SHOP_DEMANDS: [[i8; 4]; 8] = [",
    ]
    lines.extend(f"    {_rust_array(row)}," for row in data["shop_demands"])
    lines.extend([
        "];",
        "pub const TOWN_CENTER_DEMAND_SCHEDULE: [[i64; 2]; 1] = [[0, 1]];",
        f"pub const LAND_PRICES: [i32; 3] = {_rust_array(list(source.LAND_PRICES))};",
        f"pub const MARKET_I0: [f64; {len(market_order)}] = {_rust_array([market[name]['I0'] for name in market_order], _rust_float)};",
        f"pub const PRICE_FLOOR: i32 = {source.PRICE_FLOOR};",
        f"pub const MARKET_BASE: [f64; {len(market_order)}] = {_rust_array([market[name]['base'] for name in market_order], _rust_float)};",
        f"pub const MARKET_T: [f64; {len(market_order)}] = {_rust_array([market[name]['T'] for name in market_order], _rust_float)};",
        f"pub const MARKET_BELOW_TARGET: [f64; {len(market_order)}] = {_rust_array([market[name]['below_target'] for name in market_order], _rust_float)};",
        f"pub const MARKET_ABOVE_TARGET: [f64; {len(market_order)}] = {_rust_array([market[name]['above_target'] for name in market_order], _rust_float)};",
        f"pub const MARKET_BELOW_SHAPE: [u8; {len(market_order)}] = {_rust_array([{'linear': 0, 'sq': 1, 'sqrt': 2, 'log': 3, 'log10': 4, 'hinge': 5}[market[name]['below_func']] for name in market_order])};",
        f"pub const MARKET_ABOVE_SHAPE: [u8; {len(market_order)}] = {_rust_array([{'linear': 0, 'sq': 1, 'sqrt': 2, 'log': 3, 'log10': 4, 'hinge': 5}[market[name]['above_func']] for name in market_order])};",
        "pub const OBS_SIZE: usize = 5630;",
        "pub const OBS_MARKET_INVENTORY: usize = 5540;",
        "pub const OBS_MARKET_PRICES: usize = 5549;",
        "pub const OBS_SHOPS: usize = 5560;",
        "pub const OBS_SHED: usize = 5319;",
        "pub const OBS_SEEDS: usize = 5331;",
        "pub const OBS_INVENTORY: usize = 5336;",
        "pub const OBS_ANIMAL_INVENTORY: usize = 5345;",
        "pub const OBS_HAND_INVENTORY: usize = 5348;",
        "pub const OBS_HAND_POSITIONS: usize = 5280;",
        "pub const OBS_TILE_WIDTH: usize = 26;",
        "pub const OBS_FARM_WIDTH: usize = 2600;",
        "pub const NORMALIZE_MONEY: f32 = 10000.0;",
        "pub const NORMALIZE_PRICE: f32 = 1000.0;",
        "pub const NORMALIZE_COORDINATES: f32 = 9.0;",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when generated output is stale")
    args = parser.parse_args()
    rendered = render()
    current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else None
    if args.check:
        if current != rendered:
            print(f"stale generated file: {OUTPUT}")
            return 1
        return 0
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
