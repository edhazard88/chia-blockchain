from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple
from more_itertools import partition

import pytest
from blspy import AugSchemeMPL, G1Element, G2Element, PrivateKey

from chia.clvm.spend_sim import SimClient, SpendSim, sim_and_client
from chia.consensus.default_constants import DEFAULT_CONSTANTS
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.program import Program
from chia.types.blockchain_format.serialized_program import SerializedProgram
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.types.coin_spend import CoinSpend
from chia.types.condition_opcodes import ConditionOpcode
from chia.types.eligible_coin_spends import EligibleCoinSpends, UnspentLineageIds, perform_the_fast_forward
from chia.types.internal_mempool_item import InternalMempoolItem
from chia.types.mempool_inclusion_status import MempoolInclusionStatus
from chia.types.mempool_item import BundleCoinSpend
from chia.types.spend_bundle import SpendBundle
from chia.util.errors import Err
from chia.util.ints import uint64
from chia.wallet.puzzles import p2_conditions, p2_delegated_puzzle_or_hidden_puzzle
from chia.wallet.puzzles import singleton_top_layer_v1_1 as singleton_top_layer
from tests.clvm.test_puzzles import public_key_for_index, secret_exponent_for_index
from tests.core.mempool.test_mempool_manager import (
    IDENTITY_PUZZLE,
    IDENTITY_PUZZLE_HASH,
    TEST_COIN,
    TEST_COIN_ID,
    TEST_HEIGHT,
    mempool_item_from_spendbundle,
    spend_bundle_from_conditions,
)
from tests.util.key_tool import KeyTool


@pytest.mark.asyncio
async def test_process_fast_forward_spends_nothing_to_do() -> None:
    """
    This tests the case when we don't have an eligible coin, so there is
    nothing to fast forward and the item remains unchanged
    """

    async def get_unspent_lineage_ids_for_puzzle_hash(_: bytes32) -> Optional[UnspentLineageIds]:
        assert False

    conditions = [[ConditionOpcode.AGG_SIG_UNSAFE, G1Element(), IDENTITY_PUZZLE_HASH]]
    sb = spend_bundle_from_conditions(conditions, TEST_COIN)
    item = mempool_item_from_spendbundle(sb)
    # This coin is not eligible for fast forward
    assert item.bundle_coin_spends[TEST_COIN_ID].eligible_for_fast_forward is False
    internal_mempool_item = InternalMempoolItem(
        sb, item.npc_result, item.height_added_to_mempool, item.bundle_coin_spends
    )
    original_version = dataclasses.replace(internal_mempool_item)
    eligible_coin_spends = EligibleCoinSpends()
    await eligible_coin_spends.process_fast_forward_spends(
        mempool_item=internal_mempool_item,
        get_unspent_lineage_ids_for_puzzle_hash=get_unspent_lineage_ids_for_puzzle_hash,
        height=TEST_HEIGHT,
        constants=DEFAULT_CONSTANTS,
    )
    assert eligible_coin_spends == EligibleCoinSpends()
    assert internal_mempool_item == original_version


@pytest.mark.asyncio
async def test_process_fast_forward_spends_unknown_ff() -> None:
    """
    This tests the case when we process for the first time but we are unable
    to lookup the latest version from the DB
    """

    async def get_unspent_lineage_ids_for_puzzle_hash(puzzle_hash: bytes32) -> Optional[UnspentLineageIds]:
        if puzzle_hash == IDENTITY_PUZZLE_HASH:
            return None
        assert False

    test_coin = Coin(TEST_COIN_ID, IDENTITY_PUZZLE_HASH, 1)
    conditions = [[ConditionOpcode.CREATE_COIN, IDENTITY_PUZZLE_HASH, 1]]
    sb = spend_bundle_from_conditions(conditions, test_coin)
    item = mempool_item_from_spendbundle(sb)
    # The coin is eligible for fast forward
    assert item.bundle_coin_spends[test_coin.name()].eligible_for_fast_forward is True
    internal_mempool_item = InternalMempoolItem(
        sb, item.npc_result, item.height_added_to_mempool, item.bundle_coin_spends
    )
    eligible_coin_spends = EligibleCoinSpends()
    # We have no fast forward records yet, so we'll process this coin for the
    # first time here, but the DB lookup will return None
    with pytest.raises(ValueError, match="Cannot proceed with singleton spend fast forward."):
        await eligible_coin_spends.process_fast_forward_spends(
            mempool_item=internal_mempool_item,
            get_unspent_lineage_ids_for_puzzle_hash=get_unspent_lineage_ids_for_puzzle_hash,
            height=TEST_HEIGHT,
            constants=DEFAULT_CONSTANTS,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_first_time_processing_ph", [True, False])
async def test_process_fast_forward_spends_latest_unspent(is_first_time_processing_ph: bool) -> None:
    """
    This tests the case when we are the latest singleton version already, so
    we don't need to fast forward, we just need to set the next version from
    our additions to chain ff spends.
    The `is_first_time_processing_ph` parameter covers both scenarios where
    we processed the puzzle hash before or we're processing for the first time
    """
    test_coin = Coin(TEST_COIN_ID, IDENTITY_PUZZLE_HASH, 3)
    test_unspent_lineage_ids = UnspentLineageIds(
        coin_id=test_coin.name(),
        coin_amount=test_coin.amount,
        parent_id=test_coin.parent_coin_info,
        parent_amount=test_coin.amount,
        parent_parent_id=TEST_COIN_ID,
    )

    async def get_unspent_lineage_ids_for_puzzle_hash(puzzle_hash: bytes32) -> Optional[UnspentLineageIds]:
        if is_first_time_processing_ph and puzzle_hash == IDENTITY_PUZZLE_HASH:
            return test_unspent_lineage_ids
        assert False

    # At this point there is no puzzle to perform proper singleton validation
    # with, so spends are considered potentially eligible for fast forward mainly
    # when their amount is even and they don't have conditions that disqualify them
    conditions = [[ConditionOpcode.CREATE_COIN, IDENTITY_PUZZLE_HASH, 1]]
    sb = spend_bundle_from_conditions(conditions, test_coin)
    item = mempool_item_from_spendbundle(sb)
    assert item.bundle_coin_spends[test_coin.name()].eligible_for_fast_forward is True
    internal_mempool_item = InternalMempoolItem(
        sb, item.npc_result, item.height_added_to_mempool, item.bundle_coin_spends
    )
    original_version = dataclasses.replace(internal_mempool_item)
    if is_first_time_processing_ph:
        eligible_coin_spends = EligibleCoinSpends()
    else:
        eligible_coin_spends = EligibleCoinSpends(
            deduplication_spends={},
            fast_forward_spends={IDENTITY_PUZZLE_HASH: test_unspent_lineage_ids},
        )
    await eligible_coin_spends.process_fast_forward_spends(
        mempool_item=internal_mempool_item,
        get_unspent_lineage_ids_for_puzzle_hash=get_unspent_lineage_ids_for_puzzle_hash,
        height=TEST_HEIGHT,
        constants=DEFAULT_CONSTANTS,
    )
    child_coin = item.bundle_coin_spends[test_coin.name()].additions[0]
    expected_fast_forward_spends = {
        IDENTITY_PUZZLE_HASH: UnspentLineageIds(
            coin_id=child_coin.name(),
            coin_amount=child_coin.amount,
            parent_id=test_coin.name(),
            parent_amount=test_coin.amount,
            parent_parent_id=test_coin.parent_coin_info,
        )
    }
    # We have set the next version from our additions to chain ff spends
    assert eligible_coin_spends.fast_forward_spends == expected_fast_forward_spends
    # We didn't need to fast forward the item so it stays as is
    assert internal_mempool_item == original_version


def test_perform_the_fast_forward() -> None:
    """
    This test attempts to spend a coin that is already spent and the current
    unspent version is its grandchild. We fast forward the test coin spend into
    a spend of that latest unspent
    """
    test_parent_id = bytes32.from_hexstr("0x039759eda861cd44c0af6c9501300f66fe4f5de144b8ae4fc4e8da35701f38ac")
    test_ph = bytes32.from_hexstr("0x9ae0917f3ca301f934468ec60412904c0a88b232aeabf220c01ef53054e0281a")
    test_amount = 1339
    test_coin = Coin(test_parent_id, test_ph, test_amount)
    test_child_coin = Coin(test_coin.name(), test_ph, test_amount)
    latest_unspent_coin = Coin(test_child_coin.name(), test_ph, test_amount)
    # This spend setup makes us eligible for fast forward so that we perform a
    # meaningful fast forward on the rust side
    test_puzzle_reveal = SerializedProgram.fromhex(
        "ff02ffff01ff02ffff01ff02ffff03ffff18ff2fff3480ffff01ff04ffff04ff20ffff04ff2fff808080ffff04ffff02ff3effff04ff0"
        "2ffff04ff05ffff04ffff02ff2affff04ff02ffff04ff27ffff04ffff02ffff03ff77ffff01ff02ff36ffff04ff02ffff04ff09ffff04"
        "ff57ffff04ffff02ff2effff04ff02ffff04ff05ff80808080ff808080808080ffff011d80ff0180ffff04ffff02ffff03ff77ffff018"
        "1b7ffff015780ff0180ff808080808080ffff04ff77ff808080808080ffff02ff3affff04ff02ffff04ff05ffff04ffff02ff0bff5f80"
        "ffff01ff8080808080808080ffff01ff088080ff0180ffff04ffff01ffffffff4947ff0233ffff0401ff0102ffffff20ff02ffff03ff0"
        "5ffff01ff02ff32ffff04ff02ffff04ff0dffff04ffff0bff3cffff0bff34ff2480ffff0bff3cffff0bff3cffff0bff34ff2c80ff0980"
        "ffff0bff3cff0bffff0bff34ff8080808080ff8080808080ffff010b80ff0180ffff02ffff03ffff22ffff09ffff0dff0580ff2280fff"
        "f09ffff0dff0b80ff2280ffff15ff17ffff0181ff8080ffff01ff0bff05ff0bff1780ffff01ff088080ff0180ff02ffff03ff0bffff01"
        "ff02ffff03ffff02ff26ffff04ff02ffff04ff13ff80808080ffff01ff02ffff03ffff20ff1780ffff01ff02ffff03ffff09ff81b3fff"
        "f01818f80ffff01ff02ff3affff04ff02ffff04ff05ffff04ff1bffff04ff34ff808080808080ffff01ff04ffff04ff23ffff04ffff02"
        "ff36ffff04ff02ffff04ff09ffff04ff53ffff04ffff02ff2effff04ff02ffff04ff05ff80808080ff808080808080ff738080ffff02f"
        "f3affff04ff02ffff04ff05ffff04ff1bffff04ff34ff8080808080808080ff0180ffff01ff088080ff0180ffff01ff04ff13ffff02ff"
        "3affff04ff02ffff04ff05ffff04ff1bffff04ff17ff8080808080808080ff0180ffff01ff02ffff03ff17ff80ffff01ff088080ff018"
        "080ff0180ffffff02ffff03ffff09ff09ff3880ffff01ff02ffff03ffff18ff2dffff010180ffff01ff0101ff8080ff0180ff8080ff01"
        "80ff0bff3cffff0bff34ff2880ffff0bff3cffff0bff3cffff0bff34ff2c80ff0580ffff0bff3cffff02ff32ffff04ff02ffff04ff07f"
        "fff04ffff0bff34ff3480ff8080808080ffff0bff34ff8080808080ffff02ffff03ffff07ff0580ffff01ff0bffff0102ffff02ff2eff"
        "ff04ff02ffff04ff09ff80808080ffff02ff2effff04ff02ffff04ff0dff8080808080ffff01ff0bffff0101ff058080ff0180ff02fff"
        "f03ffff21ff17ffff09ff0bff158080ffff01ff04ff30ffff04ff0bff808080ffff01ff088080ff0180ff018080ffff04ffff01ffa07f"
        "aa3253bfddd1e0decb0906b2dc6247bbc4cf608f58345d173adb63e8b47c9fffa030d940e53ed5b56fee3ae46ba5f4e59da5e2cc9242f"
        "6e482fe1f1e4d9a463639a0eff07522495060c066f66f32acc2a77e3a3e737aca8baea4d1a64ea4cdc13da9ffff04ffff010dff018080"
        "80"
    )
    test_solution = SerializedProgram.fromhex(
        "ffffa030d940e53ed5b56fee3ae46ba5f4e59da5e2cc9242f6e482fe1f1e4d9a463639ffa0c7b89cfb9abf2c4cb212a4840b37d762f4c"
        "880b8517b0dadb0c310ded24dd86dff82053980ff820539ffff80ffff01ffff33ffa0c7b89cfb9abf2c4cb212a4840b37d762f4c880b8"
        "517b0dadb0c310ded24dd86dff8205398080ff808080"
    )
    test_coin_spend = CoinSpend(test_coin, test_puzzle_reveal, test_solution)
    test_spend_data = BundleCoinSpend(test_coin_spend, False, True, [test_child_coin])
    test_unspent_lineage_ids = UnspentLineageIds(
        coin_id=latest_unspent_coin.name(),
        coin_amount=latest_unspent_coin.amount,
        parent_id=latest_unspent_coin.parent_coin_info,
        parent_amount=test_child_coin.amount,
        parent_parent_id=test_child_coin.parent_coin_info,
    )
    # Start from a fresh state of fast forward spends
    fast_forward_spends: Dict[bytes32, UnspentLineageIds] = {}
    # Perform the fast forward on the test coin (the grandparent)
    new_coin_spend, patched_additions = perform_the_fast_forward(
        test_unspent_lineage_ids, test_spend_data, fast_forward_spends
    )
    # Make sure the new coin we got is the grandchild (latest unspent version)
    assert new_coin_spend.coin == latest_unspent_coin
    # Make sure the puzzle reveal is intact
    assert new_coin_spend.puzzle_reveal == test_coin_spend.puzzle_reveal
    # Make sure the solution got patched
    assert new_coin_spend.solution != test_coin_spend.solution
    # Make sure the additions got patched
    expected_child_coin = Coin(latest_unspent_coin.name(), test_ph, test_amount)
    assert patched_additions == [expected_child_coin]
    # Make sure the new fast forward state got updated with the latest unspent
    # becoming the new child, with its parent being the version we just spent
    # (previously latest unspent)
    expected_unspent_lineage_ids = UnspentLineageIds(
        coin_id=expected_child_coin.name(),
        coin_amount=expected_child_coin.amount,
        parent_id=latest_unspent_coin.name(),
        parent_amount=latest_unspent_coin.amount,
        parent_parent_id=latest_unspent_coin.parent_coin_info,
    )
    assert fast_forward_spends == {test_ph: expected_unspent_lineage_ids}


def sign_delegated_puz(del_puz: Program, coin: Coin) -> G2Element:
    synthetic_secret_key: PrivateKey = p2_delegated_puzzle_or_hidden_puzzle.calculate_synthetic_secret_key(
        PrivateKey.from_bytes(secret_exponent_for_index(1).to_bytes(32, "big")),
        p2_delegated_puzzle_or_hidden_puzzle.DEFAULT_HIDDEN_PUZZLE_HASH,
    )
    return AugSchemeMPL.sign(
        synthetic_secret_key, (del_puz.get_tree_hash() + coin.name() + DEFAULT_CONSTANTS.AGG_SIG_ME_ADDITIONAL_DATA)
    )


async def make_and_send_spend_bundle(
    sim: SpendSim,
    sim_client: SimClient,
    coin: Coin,
    delegated_puzzle: Program,
    coin_spends: List[CoinSpend],
    is_eligible_for_ff: bool,
    is_launcher_coin: bool = False,
    farm_afterwards: bool = True,
) -> Tuple[MempoolInclusionStatus, Optional[Err]]:
    if is_launcher_coin or not is_eligible_for_ff:
        print("is_launcher_coin: ", is_launcher_coin)
        print("is_eligible_for_ff: ", is_eligible_for_ff)
        signature = sign_delegated_puz(delegated_puzzle, coin)
        print("signature: ", signature)
    else:
        signature = G2Element()
    spend_bundle = SpendBundle(coin_spends, signature)
    status, error = await sim_client.push_tx(spend_bundle)
    if error is None and farm_afterwards:
        await sim.farm_block()
    return status, error


async def get_singleton_and_remaining_coins(sim: SpendSim) -> Tuple[Coin, List[Coin]]:
    coins = await sim.all_non_reward_coins()
    singletons, remaining_coins = partition(lambda coin: coin.amount % 2 == 0, coins)
    singletons_list = list(singletons)
    assert len(singletons_list) == 1
    return singletons_list[0], list(remaining_coins)


def make_singleton_coin_spend(
    parent_coin_spend: CoinSpend,
    coin_to_spend: Coin,
    inner_puzzle: Program,
    inner_conditions: List[List[Any]],
    is_eve_spend: bool = False,
) -> CoinSpend:
    lineage_proof = singleton_top_layer.lineage_proof_for_coinsol(parent_coin_spend)
    delegated_puzzle = Program.to((1, inner_conditions))
    inner_solution = Program.to([[], delegated_puzzle, []])
    solution = singleton_top_layer.solution_for_singleton(lineage_proof, uint64(coin_to_spend.amount), inner_solution)
    if is_eve_spend:
        # Parent here is the launcher coin
        puzzle_reveal = SerializedProgram.from_program(
            singleton_top_layer.puzzle_for_singleton(parent_coin_spend.coin.name(), inner_puzzle)
        )
    else:
        puzzle_reveal = parent_coin_spend.puzzle_reveal
    return CoinSpend(coin_to_spend, puzzle_reveal, solution)


async def prepare_singleton_eve(
    sim: SpendSim, sim_client: SimClient, is_eligible_for_ff: bool, start_amount: uint64, singleton_amount: uint64
) -> Tuple[Program, CoinSpend, Program]:
    # Generate starting info
    key_lookup = KeyTool()
    pk = G1Element.from_bytes(public_key_for_index(1, key_lookup))
    starting_puzzle = p2_delegated_puzzle_or_hidden_puzzle.puzzle_for_pk(pk)
    if is_eligible_for_ff:
        # This program allows us to control conditions through solutions
        inner_puzzle = Program.to(13)
    else:
        inner_puzzle = starting_puzzle
    inner_puzzle_hash = inner_puzzle.get_tree_hash()
    # Get our starting standard coin created
    await sim.farm_block(starting_puzzle.get_tree_hash())
    records = await sim_client.get_coin_records_by_puzzle_hash(starting_puzzle.get_tree_hash())
    starting_coin = records[0].coin
    print("starting_coin amount: ", starting_coin.amount)
    # Launching
    conditions, launcher_coin_spend = singleton_top_layer.launch_conditions_and_coinsol(
        coin=starting_coin, inner_puzzle=inner_puzzle, comment=[], amount=start_amount
    )
    # Keep a remaining coin with an even amount
    # conditions.append(
    #     Program.to([ConditionOpcode.CREATE_COIN, IDENTITY_PUZZLE_HASH, starting_coin.amount - start_amount - 1])
    # )
    # Create a solution for standard transaction
    delegated_puzzle = p2_conditions.puzzle_for_conditions(conditions)
    full_solution = p2_delegated_puzzle_or_hidden_puzzle.solution_for_conditions(conditions)
    starting_coin_spend = CoinSpend(starting_coin, starting_puzzle, full_solution)
    await make_and_send_spend_bundle(
        sim,
        sim_client,
        starting_coin,
        delegated_puzzle,
        [starting_coin_spend, launcher_coin_spend],
        is_eligible_for_ff,
        is_launcher_coin=True,
    )
    # Eve coin
    # eve_coin, [remaining_coin] = await get_singleton_and_remaining_coins(sim)
    eve_coin, _ = await get_singleton_and_remaining_coins(sim)
    # print("remaining_coin amount: ", remaining_coin.amount)
    print("eve_coin.amount: ", eve_coin.amount)
    inner_conditions = [[ConditionOpcode.CREATE_COIN, inner_puzzle_hash, singleton_amount]]
    singleton_eve_coin_spend = make_singleton_coin_spend(
        parent_coin_spend=launcher_coin_spend,
        coin_to_spend=eve_coin,
        inner_puzzle=starting_puzzle,
        inner_conditions=inner_conditions,
        is_eve_spend=True,
    )
    status, error = await make_and_send_spend_bundle(
        sim, sim_client, eve_coin, delegated_puzzle, [singleton_eve_coin_spend], is_eligible_for_ff
    )
    print("status, error: ", (status, error))
    exit()
    return starting_puzzle, singleton_eve_coin_spend, delegated_puzzle


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eligible_for_ff", [True, False])
async def test_singleton_fast_forward_different_block(is_eligible_for_ff: bool) -> None:
    START_AMOUNT = uint64(1337)
    # We're decrementing the next iteration's amount for testing purposes
    SINGLETON_AMOUNT = uint64(1335)
    # We're incrementing the next iteration's amount for testing purposes
    SINGLETON_CHILD_AMOUNT = uint64(1339)
    async with sim_and_client() as (sim, sim_client):
        starting_puzzle, singleton_eve_coin_spend, delegated_puzzle = await prepare_singleton_eve(
            sim, sim_client, is_eligible_for_ff, START_AMOUNT, SINGLETON_AMOUNT
        )
        singleton_puzzle_hash = singleton_eve_coin_spend.coin.puzzle_hash
        # At this point we don't have any unspent singleton
        unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
            singleton_puzzle_hash
        )
        assert unspent_lineage_ids is None
        singleton_eve = singleton_eve_coin_spend.coin
        print("foo")
        await make_and_send_spend_bundle(
            sim, sim_client, singleton_eve, delegated_puzzle, [singleton_eve_coin_spend], is_eligible_for_ff
        )
        print("bar")
        # Now we spent eve and we have an unspent singleton that we can test with
        singleton, [remaining_coin] = await get_singleton_and_remaining_coins(sim)
        assert singleton.amount == SINGLETON_AMOUNT
        print("singleton amount: ", singleton.amount)
        unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
            singleton_puzzle_hash
        )
        assert unspent_lineage_ids == UnspentLineageIds(
            coin_id=singleton.name(),
            coin_amount=singleton.amount,
            parent_id=singleton_eve.name(),
            parent_amount=singleton_eve.amount,
            parent_parent_id=singleton_eve.parent_coin_info,
        )
        # Let's spend this first version, to create a bigger singleton child
        inner_conditions = [
            [ConditionOpcode.AGG_SIG_UNSAFE, G1Element(), IDENTITY_PUZZLE_HASH],
            [ConditionOpcode.CREATE_COIN, starting_puzzle.get_tree_hash(), SINGLETON_CHILD_AMOUNT],
        ]
        singleton_coin_spend = make_singleton_coin_spend(
            singleton_eve_coin_spend, singleton, starting_puzzle, inner_conditions
        )
        # Spend also a remaining coin for balance, as we're increasing the singleton amount
        diff_to_balance = SINGLETON_CHILD_AMOUNT - SINGLETON_AMOUNT
        remaining_spend_solution = SerializedProgram.from_program(
            Program.to([[ConditionOpcode.CREATE_COIN, IDENTITY_PUZZLE_HASH, remaining_coin.amount - diff_to_balance]])
        )
        remaining_coin_spend = CoinSpend(remaining_coin, IDENTITY_PUZZLE, remaining_spend_solution)
        status, error = await make_and_send_spend_bundle(
            sim,
            sim_client,
            singleton,
            delegated_puzzle,
            [remaining_coin_spend, singleton_coin_spend],
            is_eligible_for_ff,
        )
        print("status, error: ", (status, error))
        unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
            singleton_puzzle_hash
        )
        singleton_child, [remaining_coin] = await get_singleton_and_remaining_coins(sim)
        print("remaining_coin.amount: ", remaining_coin.amount)
        print("singleton_child.amount: ", singleton_child.amount)
        assert singleton_child.amount == SINGLETON_CHILD_AMOUNT
        assert unspent_lineage_ids == UnspentLineageIds(
            coin_id=singleton_child.name(),
            coin_amount=singleton_child.amount,
            parent_id=singleton.name(),
            parent_amount=singleton.amount,
            parent_parent_id=singleton_eve.name(),
        )
        # Now let's spend the first version again (despite being already spent by now)
        remaining_coin_spend = CoinSpend(
            remaining_coin,
            IDENTITY_PUZZLE,
            Program.to([[ConditionOpcode.CREATE_COIN, IDENTITY_PUZZLE_HASH, remaining_coin.amount - 10000]]),
        )
        status, error = await make_and_send_spend_bundle(
            sim,
            sim_client,
            singleton,
            delegated_puzzle,
            [remaining_coin_spend, singleton_coin_spend],
            is_eligible_for_ff,
        )
        if is_eligible_for_ff:
            # Instead of rejecting this as double spend, we perform a fast forward,
            # spending the singleton child as a result, and creating the latest
            # version which is the grandchild in this scenario
            assert status == MempoolInclusionStatus.SUCCESS
            assert error is None
            unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
                singleton_puzzle_hash
            )
            singleton_grandchild, [remaining_coin] = await get_singleton_and_remaining_coins(sim)
            print("remaining_coin amount: ", remaining_coin.amount)
            print("singleton_grandchild amount: ", singleton_grandchild.amount)
            print("singleton_child.amount: ", singleton_child.amount)
            assert unspent_lineage_ids == UnspentLineageIds(
                coin_id=singleton_grandchild.name(),
                coin_amount=singleton_grandchild.amount,
                parent_id=singleton_child.name(),
                parent_amount=singleton_child.amount,
                parent_parent_id=singleton.name(),
            )
            print("-- Coin IDs:")
            print("eve: ", singleton_eve.name().hex())
            print("singleton: ", singleton.name().hex())
            print("singleton_child: ", singleton_child.name().hex())
            print("singleton_grandchild: ", singleton_grandchild.name().hex())
        else:
            # As this singleton is not eligible for fast forward, attempting to
            # spend one of its earlier versions is considered a double spend
            assert status == MempoolInclusionStatus.FAILED
            assert error == Err.DOUBLE_SPEND


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eligible_for_ff", [True, False])
async def test_singleton_fast_forward(is_eligible_for_ff: bool) -> None:
    START_AMOUNT = uint64(1337)
    # We're decrementing the next iteration's amount for testing purposes
    SINGLETON_AMOUNT = uint64(1335)
    # We're incrementing the next iteration's amount for testing purposes
    async with sim_and_client() as (sim, sim_client):
        starting_puzzle, singleton_eve_coin_spend, delegated_puzzle = await prepare_singleton_eve(
            sim, sim_client, is_eligible_for_ff, START_AMOUNT, SINGLETON_AMOUNT
        )
        singleton_puzzle_hash = singleton_eve_coin_spend.coin.puzzle_hash
        # At this point we don't have any unspent singleton
        unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
            singleton_puzzle_hash
        )
        assert unspent_lineage_ids is None
        singleton_eve = singleton_eve_coin_spend.coin
        await make_and_send_spend_bundle(
            sim, sim_client, singleton_eve, delegated_puzzle, [singleton_eve_coin_spend], is_eligible_for_ff
        )
        # Now we spent eve and we have an unspent singleton that we can test with
        singleton, [remaining_coin] = await get_singleton_and_remaining_coins(sim)
        print("remaining_coin amount: ", remaining_coin.amount)
        print("singleton amount: ", singleton.amount)
        unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
            singleton_puzzle_hash
        )
        assert singleton.amount == SINGLETON_AMOUNT
        assert unspent_lineage_ids == UnspentLineageIds(
            coin_id=singleton.name(),
            coin_amount=singleton.amount,
            parent_id=singleton_eve.name(),
            parent_amount=singleton_eve.amount,
            parent_parent_id=singleton_eve.parent_coin_info,
        )
        # Let's spend this first version, to create singleton child
        # Same delegated_puzzle/inner_solution as we're just recreating ourselves here
        lineage_proof = singleton_top_layer.lineage_proof_for_coinsol(singleton_eve_coin_spend)

        delegated_puzzle2 = Program.to(
            (
                1,
                [
                    [ConditionOpcode.AGG_SIG_UNSAFE, G1Element(), IDENTITY_PUZZLE_HASH],
                    [ConditionOpcode.CREATE_COIN, starting_puzzle.get_tree_hash(), 1339],
                ],
            )
        )
        inner_solution2 = Program.to([[], delegated_puzzle2, []])
        solution2 = singleton_top_layer.solution_for_singleton(lineage_proof, uint64(singleton.amount), inner_solution2)

        # inner_solution = Program.to([[], delegated_puzzle, []])
        # full_solution = singleton_top_layer.solution_for_singleton(
        #     lineage_proof, uint64(singleton.amount), inner_solution
        # )
        # Same puzzle reveal too
        # singleton_coin_spend = CoinSpend(singleton, singleton_eve_coin_spend.puzzle_reveal, full_solution)
        singleton_coin_spend = CoinSpend(singleton, singleton_eve_coin_spend.puzzle_reveal, solution2)
        remaining_coin_spend = CoinSpend(
            remaining_coin,
            IDENTITY_PUZZLE,
            Program.to([[ConditionOpcode.CREATE_COIN, IDENTITY_PUZZLE_HASH, remaining_coin.amount - 10000]]),
        )
        # print("full_solution: ", full_solution)

        # ===============================================================

        # 1) Generate like 5 spend bundles that all spend the singleton and output a new singleton with a unique amount, and light an XCH coin with a large amount on fire
        # 2) Shuffle them (or test every combination) and send them to the mempool
        # 3) Examine the history and make sure the coin amounts went in the order that the spend bundles you submitted implies
        # inner_puzzle_hash = Program.to(13).get_tree_hash()
        # delegated_puzzle1 = Program.to(
        #     (
        #         1,
        #         [
        #             [ConditionOpcode.AGG_SIG_UNSAFE, G1Element(), IDENTITY_PUZZLE_HASH],
        #             [ConditionOpcode.CREATE_COIN, inner_puzzle_hash, 1335],
        #         ],
        #     )
        # )
        # inner_solution1 = Program.to([[], delegated_puzzle1, []])
        # solution1 = singleton_top_layer.solution_for_singleton(lineage_proof, uint64(singleton.amount), inner_solution1)
        # spend1 = CoinSpend(singleton, singleton_eve_coin_spend.puzzle_reveal, solution1)
        # delegated_puzzle2 = Program.to(
        #     (
        #         1,
        #         [
        #             [ConditionOpcode.AGG_SIG_UNSAFE, G1Element(), IDENTITY_PUZZLE_HASH],
        #             [ConditionOpcode.CREATE_COIN, inner_puzzle_hash, 1333],
        #         ],
        #     )
        # )
        # inner_solution2 = Program.to([[], delegated_puzzle2, []])
        # solution2 = singleton_top_layer.solution_for_singleton(lineage_proof, uint64(singleton.amount), inner_solution2)
        # spend2 = CoinSpend(singleton, singleton_eve_coin_spend.puzzle_reveal, solution2)
        # # We spend the singleton and get its child as the most recent unspent
        # print("mempool items before sending spend1: ", len(list(sim.mempool_manager.mempool.all_items())))
        # status, error = await make_and_send_spend_bundle(
        #     sim, sim_client, singleton, delegated_puzzle1, [spend1], is_eligible_for_ff, farm_afterwards=False
        # )
        # print("status, error: ", (status, error))
        # print("mempool items before farming spend1: ", len(list(sim.mempool_manager.mempool.all_items())))
        # print("farming after spend1")
        # await sim.farm_block()
        # print("done farming spend1")
        # print("mempool items after farming spend1: ", len(list(sim.mempool_manager.mempool.all_items())))
        # # print([x.name.hex() for x in sim.mempool_manager.mempool.all_items()])
        # singleton_child = (await sim.all_non_reward_coins())[0]
        # print("singleton_child: ", singleton_child.name().hex())
        # print("singleton_child.amount: ", singleton_child.amount)
        # status, error = await make_and_send_spend_bundle(
        #     sim, sim_client, singleton, delegated_puzzle1, [spend2], is_eligible_for_ff, farm_afterwards=False
        # )
        # print("status, error: ", (status, error))
        # print("mempool items before farming spend2: ", len(list(sim.mempool_manager.mempool.all_items())))
        # print("farming after spend2")
        # await sim.farm_block()
        # print("done farming spend2")
        # print("mempool items after farming spend2: ", len(list(sim.mempool_manager.mempool.all_items())))
        # unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
        #     singleton_puzzle_hash
        # )
        # singleton_grandchild = (await sim.all_non_reward_coins())[0]
        # print("singleton_grandchild: ", singleton_grandchild.name().hex())
        # print("singleton_grandchild.amount: ", singleton_grandchild.amount)
        # assert unspent_lineage_ids == UnspentLineageIds(
        #     coin_id=singleton_grandchild.name(),
        #     coin_amount=singleton_grandchild.amount,
        #     parent_id=singleton_child.name(),
        #     parent_amount=singleton_child.amount,
        #     parent_parent_id=singleton_child.parent_coin_info,
        # )
        # return

        # ===============================================================

        status, error = await make_and_send_spend_bundle(
            sim,
            sim_client,
            singleton,
            delegated_puzzle,
            [remaining_coin_spend, singleton_coin_spend],
            is_eligible_for_ff,
        )
        print("status, error: ", (status, error))
        unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
            singleton_puzzle_hash
        )
        singleton_child, [remaining_coin] = await get_singleton_and_remaining_coins(sim)
        print("remaining_coin.amount: ", remaining_coin.amount)
        print("singleton_child.amount: ", singleton_child.amount)
        assert unspent_lineage_ids == UnspentLineageIds(
            coin_id=singleton_child.name(),
            coin_amount=singleton_child.amount,
            parent_id=singleton.name(),
            parent_amount=singleton.amount,
            parent_parent_id=singleton_eve.name(),
        )
        # Now let's spend the first version again (despite being already spent by now)
        remaining_coin_spend = CoinSpend(
            remaining_coin,
            IDENTITY_PUZZLE,
            Program.to([[ConditionOpcode.CREATE_COIN, IDENTITY_PUZZLE_HASH, remaining_coin.amount - 10000]]),
        )
        status, error = await make_and_send_spend_bundle(
            sim,
            sim_client,
            singleton,
            delegated_puzzle,
            [remaining_coin_spend, singleton_coin_spend],
            is_eligible_for_ff,
        )
        if is_eligible_for_ff:
            # Instead of rejecting this as double spend, we perform a fast forward,
            # spending the singleton child as a result, and creating the latest
            # version which is the grandchild in this scenario
            assert status == MempoolInclusionStatus.SUCCESS
            assert error is None
            unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
                singleton_puzzle_hash
            )
            singleton_grandchild, [remaining_coin] = await get_singleton_and_remaining_coins(sim)
            print("remaining_coin amount: ", remaining_coin.amount)
            print("singleton_grandchild amount: ", singleton_grandchild.amount)
            print("singleton_child.amount: ", singleton_child.amount)
            assert unspent_lineage_ids == UnspentLineageIds(
                coin_id=singleton_grandchild.name(),
                coin_amount=singleton_grandchild.amount,
                parent_id=singleton_child.name(),
                parent_amount=singleton_child.amount,
                parent_parent_id=singleton.name(),
            )
            print("-- Coin IDs:")
            print("eve: ", singleton_eve.name().hex())
            print("singleton: ", singleton.name().hex())
            print("singleton_child: ", singleton_child.name().hex())
            print("singleton_grandchild: ", singleton_grandchild.name().hex())
            print("###### Now let's go further: ######")
            # Now let's go further:
            # 1) Generate like 5 spend bundles that all spend the singleton and output a new singleton with a unique amount, and light an XCH coin with a large amount on fire
            # 2) Shuffle them (or test every combination) and send them to the mempool
            # 3) Examine the history and make sure the coin amounts went in the order that the spend bundles you submitted implies
            remaining_coin_spend = CoinSpend(
                remaining_coin,
                IDENTITY_PUZZLE,
                Program.to([[ConditionOpcode.CREATE_COIN, IDENTITY_PUZZLE_HASH, remaining_coin.amount - 10000]]),
            )
            random_amounts = [21, 17, 11]
            # coin_spends = []
            signature = G2Element()
            remaining_coin_sb = SpendBundle([remaining_coin_spend], signature)
            for i in range(3):
                # this allows us to maintain the order of spend as their fee per
                # cost gets smaller due to their amounts
                cost_factor = (i + 1) * 5
                conditions = [
                    [ConditionOpcode.AGG_SIG_UNSAFE, G1Element(), IDENTITY_PUZZLE_HASH] for _ in range(cost_factor)
                ]
                conditions.append([ConditionOpcode.CREATE_COIN, starting_puzzle.get_tree_hash(), random_amounts[i]])
                delegated_puzzle = Program.to((1, conditions))
                inner_solution = Program.to([[], delegated_puzzle, []])
                solution = singleton_top_layer.solution_for_singleton(
                    lineage_proof, uint64(singleton.amount), inner_solution
                )
                singleton_coin_spend = CoinSpend(singleton, singleton_eve_coin_spend.puzzle_reveal, solution)
                print("about to call push tx on item ")
                coin_spends = [singleton_coin_spend]
                # if i == 1:
                #     coin_spends.append(remaining_coin_spend)
                sb = SpendBundle(coin_spends, signature)
                status, error = await sim_client.push_tx(sb)
                print("status, error: ", (status, error))
                print("item ", i, " has amount ", random_amounts[i], " and id ", sb.name().hex())
                # coin_spends.append(singleton_coin_spend)
            # coin_spends.append(remaining_coin_spend)
            # coin_spends_sb = SpendBundle(coin_spends, signature)
            # sb = SpendBundle.aggregate([coin_spends_sb, remaining_coin_sb])
            # print("about to call push tx on remaining")
            # status, error = await sim_client.push_tx(remaining_coin_sb)
            # status, error = await make_and_send_spend_bundle(
            #     sim, sim_client, singleton, delegated_puzzle, coin_spends, is_eligible_for_ff, farm_afterwards=False
            # )
            # print("status, error: ", (status, error))
            print("=========== now we farm ===========")
            await sim.farm_block()
            unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
                singleton_puzzle_hash
            )
            latest_singleton, [remaining_coin] = await get_singleton_and_remaining_coins(sim)
            print("remaining_coin amount: ", remaining_coin.amount)
            print("latest_singleton amount: ", latest_singleton.amount)
            assert unspent_lineage_ids is not None
            assert unspent_lineage_ids.coin_id == latest_singleton.name()
            assert latest_singleton.amount == random_amounts[-1]
            assert unspent_lineage_ids.coin_amount == latest_singleton.amount
            assert unspent_lineage_ids.parent_id == latest_singleton.parent_coin_info
            assert unspent_lineage_ids.parent_amount == random_amounts[-2]
            print("-- Coin IDs:")
            print("eve: ", singleton_eve.name().hex())
            print("singleton: ", singleton.name().hex())
            print("singleton_child: ", singleton_child.name().hex())
            print("singleton_grandchild: ", singleton_grandchild.name().hex())
        else:
            # As this singleton is not eligible for fast forward, attempting to
            # spend one of its earlier versions is considered a double spend
            assert status == MempoolInclusionStatus.FAILED
            assert error == Err.DOUBLE_SPEND


async def create_singleton_eve(
    sim: SpendSim, sim_client: SimClient, is_eligible_for_ff: bool
) -> Tuple[CoinSpend, Program, bytes32]:
    # Generate starting info
    key_lookup = KeyTool()
    pk = G1Element.from_bytes(public_key_for_index(1, key_lookup))
    starting_puzzle = p2_delegated_puzzle_or_hidden_puzzle.puzzle_for_pk(pk)
    if is_eligible_for_ff:
        inner_puzzle = Program.to(13)
    else:
        inner_puzzle = starting_puzzle
    inner_puzzle_hash = inner_puzzle.get_tree_hash()
    # Get our starting standard coin created
    START_AMOUNT = uint64(1337)
    await sim.farm_block(starting_puzzle.get_tree_hash())
    records = await sim_client.get_coin_records_by_puzzle_hash(starting_puzzle.get_tree_hash())
    starting_coin = records[0].coin
    # Launching
    conditions, launcher_coin_spend = singleton_top_layer.launch_conditions_and_coinsol(
        coin=starting_coin, inner_puzzle=inner_puzzle, comment=[], amount=START_AMOUNT
    )
    # Creating solution for standard transaction
    delegated_puzzle = p2_conditions.puzzle_for_conditions(conditions)
    full_solution = p2_delegated_puzzle_or_hidden_puzzle.solution_for_conditions(conditions)
    starting_coin_spend = CoinSpend(starting_coin, starting_puzzle, full_solution)
    await make_and_send_spend_bundle(
        sim,
        sim_client,
        starting_coin,
        delegated_puzzle,
        [starting_coin_spend, launcher_coin_spend],
        is_eligible_for_ff,
        is_launcher_coin=True,
    )
    # Eve coin
    eve_coin = (await sim.all_non_reward_coins())[0]
    singleton_puzzle_hash = eve_coin.puzzle_hash
    launcher_coin = singleton_top_layer.generate_launcher_coin(starting_coin, START_AMOUNT)
    launcher_id = launcher_coin.name()
    # This delegated puzzle just recreates the coin exactly
    delegated_puzzle = Program.to((1, [[ConditionOpcode.CREATE_COIN, inner_puzzle_hash, eve_coin.amount]]))
    inner_solution = Program.to([[], delegated_puzzle, []])
    # Generate the lineage proof we will need from the launcher coin
    lineage_proof = singleton_top_layer.lineage_proof_for_coinsol(launcher_coin_spend)
    puzzle_reveal = singleton_top_layer.puzzle_for_singleton(launcher_id, inner_puzzle)
    full_solution = singleton_top_layer.solution_for_singleton(lineage_proof, uint64(eve_coin.amount), inner_solution)
    singleton_eve_coin_spend = CoinSpend(eve_coin, puzzle_reveal, full_solution)
    return singleton_eve_coin_spend, delegated_puzzle, singleton_puzzle_hash


@pytest.mark.asyncio
@pytest.mark.parametrize("is_eligible_for_ff", [True, False])
async def test_foo(is_eligible_for_ff: bool) -> None:
    async with sim_and_client() as (sim, sim_client):
        singleton_eve_coin_spend, delegated_puzzle, singleton_puzzle_hash = await create_singleton_eve(
            sim, sim_client, is_eligible_for_ff
        )
        # At this point we don't have any unspent singleton
        unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
            singleton_puzzle_hash
        )
        assert unspent_lineage_ids is None
        singleton_eve = singleton_eve_coin_spend.coin
        await make_and_send_spend_bundle(
            sim, sim_client, singleton_eve, delegated_puzzle, [singleton_eve_coin_spend], is_eligible_for_ff
        )
        # Now we spent eve and we have an unspent singleton that we can test with
        singleton = (await sim.all_non_reward_coins())[0]
        unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
            singleton_puzzle_hash
        )
        assert unspent_lineage_ids == UnspentLineageIds(
            coin_id=singleton.name(),
            coin_amount=singleton.amount,
            parent_id=singleton_eve.name(),
            parent_amount=singleton_eve.amount,
            parent_parent_id=singleton_eve.parent_coin_info,
        )
        # Let's spend this first version, to create singleton child
        # Same delegated_puzzle/inner_solution as we're just recreating ourselves here
        lineage_proof = singleton_top_layer.lineage_proof_for_coinsol(singleton_eve_coin_spend)
        inner_solution = Program.to([[], delegated_puzzle, []])
        full_solution = singleton_top_layer.solution_for_singleton(
            lineage_proof, uint64(singleton.amount), inner_solution
        )
        # Same puzzle reveal too
        singleton_coin_spend = CoinSpend(singleton, singleton_eve_coin_spend.puzzle_reveal, full_solution)
        # We spend the singleton and get its child as the most recent unspent
        await make_and_send_spend_bundle(
            sim, sim_client, singleton, delegated_puzzle, [singleton_coin_spend], is_eligible_for_ff
        )
        unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
            singleton_puzzle_hash
        )
        singleton_child = (await sim.all_non_reward_coins())[0]
        assert unspent_lineage_ids == UnspentLineageIds(
            coin_id=singleton_child.name(),
            coin_amount=singleton_child.amount,
            parent_id=singleton.name(),
            parent_amount=singleton.amount,
            parent_parent_id=singleton_eve.name(),
        )
        # Now let's spend the first version again (despite being already spent by now)
        status, error = await make_and_send_spend_bundle(
            sim, sim_client, singleton, delegated_puzzle, [singleton_coin_spend], is_eligible_for_ff
        )
        if is_eligible_for_ff:
            # Instead of rejecting this as double spend, we perform a fast forward,
            # spending the singleton child as a result, and creating the latest
            # version which is the grandchild in this scenario
            assert status == MempoolInclusionStatus.SUCCESS
            assert error is None
            unspent_lineage_ids = await sim_client.service.coin_store.get_unspent_lineage_ids_for_puzzle_hash(
                singleton_puzzle_hash
            )
            singleton_grandchild = (await sim.all_non_reward_coins())[0]
            assert unspent_lineage_ids == UnspentLineageIds(
                coin_id=singleton_grandchild.name(),
                coin_amount=singleton_grandchild.amount,
                parent_id=singleton_child.name(),
                parent_amount=singleton_grandchild.amount,
                parent_parent_id=singleton.name(),
            )
        else:
            # As this singleton is not eligible for fast forward, attempting to
            # spend one of its earlier versions is considered a double spend
            assert status == MempoolInclusionStatus.FAILED
            assert error == Err.DOUBLE_SPEND
