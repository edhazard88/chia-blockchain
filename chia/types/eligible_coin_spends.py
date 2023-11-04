from __future__ import annotations

import dataclasses
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from chia_rs import CoinSpend as RustCoinSpend
from chia_rs import Program as RustProgram
from chia_rs import fast_forward_singleton

from chia.consensus.condition_costs import ConditionCost
from chia.consensus.constants import ConsensusConstants
from chia.full_node.bundle_tools import simple_solution_generator
from chia.full_node.mempool_check_conditions import get_name_puzzle_conditions
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.serialized_program import SerializedProgram
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.types.coin_spend import CoinSpend
from chia.types.internal_mempool_item import InternalMempoolItem
from chia.types.mempool_item import BundleCoinSpend
from chia.types.spend_bundle import SpendBundle
from chia.util.ints import uint32, uint64


def run_for_cost(
    puzzle_reveal: SerializedProgram, solution: SerializedProgram, additions_count: int, max_cost: int
) -> uint64:
    create_coins_cost = additions_count * ConditionCost.CREATE_COIN.value
    clvm_cost, _ = puzzle_reveal.run_mempool_with_cost(max_cost, solution)
    saved_cost = uint64(clvm_cost + create_coins_cost)
    return saved_cost


@dataclasses.dataclass(frozen=True)
class DedupCoinSpend:
    solution: SerializedProgram
    cost: Optional[uint64]


@dataclasses.dataclass(frozen=True)
class UnspentLineageIds:
    coin_id: bytes32
    coin_amount: int
    parent_id: bytes32
    parent_amount: int
    parent_parent_id: bytes32


def set_next_singleton_version(
    current_singleton: Coin, singleton_additions: List[Coin], fast_forward_spends: Dict[bytes32, UnspentLineageIds]
) -> None:
    """
    Finds the next version of the singleton among its additions and updates the
    fast forward spends recrod accordingly

    Args:
        current_singleton: the current iteration of the singleton
        singleton_additions: the additions of the current singleton
        fast_forward_spends: the fast forward spends recrod

    Raises:
        ValueError if none of the additions are considered to be the singleton's
        next iteration
    """
    singleton_child = next(
        (addition for addition in singleton_additions if addition.puzzle_hash == current_singleton.puzzle_hash), None
    )
    if singleton_child is None:
        raise ValueError("Could not find fast forward child singleton.")
    # Keep track of this in order to chain the next ff
    fast_forward_spends[current_singleton.puzzle_hash] = UnspentLineageIds(
        coin_id=singleton_child.name(),
        coin_amount=singleton_child.amount,
        parent_id=singleton_child.parent_coin_info,
        parent_amount=current_singleton.amount,
        parent_parent_id=current_singleton.parent_coin_info,
    )


def perform_the_fast_forward(
    unspent_lineage_ids: UnspentLineageIds,
    spend_data: BundleCoinSpend,
    fast_forward_spends: Dict[bytes32, UnspentLineageIds],
) -> Tuple[CoinSpend, List[Coin]]:
    """
    Performs a singleton fast forward, including the updating of all previous
    additions to point to the most recent version, and updates the fast forward
    spends recrod accordingly

    Args:
        unspent_lineage_ids: the singleton's most recent ID and its parent's parent ID
        spend_data: the current spend's data
        fast_forward_spends: the fast forward spends recrod

    Returns:
        CoinSpend: the new coin spend after performing the fast forward
        List[Coin]: the updated additions that point to the new coin to spend

    Raises:
        ValueError if none of the additions are considered to be the singleton's
        next iteration
    """
    print("Entering perform_the_fast_forward")
    new_coin = Coin(
        unspent_lineage_ids.parent_id,
        spend_data.coin_spend.coin.puzzle_hash,
        unspent_lineage_ids.coin_amount,
    )
    new_parent = Coin(
        unspent_lineage_ids.parent_parent_id,
        spend_data.coin_spend.coin.puzzle_hash,
        unspent_lineage_ids.parent_amount,
    )
    # These hold because puzzle hash and amount are not expected to change
    assert new_coin.name() == unspent_lineage_ids.coin_id
    print("unspent_lineage_ids.coin_id: ", unspent_lineage_ids.coin_id.hex())
    print("assert1 good")
    assert new_parent.name() == unspent_lineage_ids.parent_id
    print("new_parent.name(): ", new_parent.name().hex())
    print("assert2 good")
    rust_coin_spend = RustCoinSpend(
        coin=spend_data.coin_spend.coin,
        puzzle_reveal=RustProgram.from_bytes(bytes(spend_data.coin_spend.puzzle_reveal)),
        solution=RustProgram.from_bytes(bytes(spend_data.coin_spend.solution)),
    )
    new_solution = SerializedProgram.from_bytes(
        fast_forward_singleton(spend=rust_coin_spend, new_coin=new_coin, new_parent=new_parent)
    )
    # print("old solution: ", spend_data.coin_spend.solution)
    print("OLD SOLUTION DATA ==============")
    new_lineage_proof, new_my_amount, new_inner_solution = spend_data.coin_spend.solution.to_program().as_python()
    parent_parent_coin_info, parent_inner_puzzle_hash, parent_amount = new_lineage_proof
    from clvm.casts import int_from_bytes
    print("parent_amount: ", int_from_bytes(parent_amount))
    print("parent_inner_puzzle_hash: ", parent_inner_puzzle_hash.hex())
    print("parent_parent_coin_info: ", parent_parent_coin_info.hex())
    print("new_my_amount: ", int_from_bytes(new_my_amount))
    _, delegated_puzzle, _ = new_inner_solution
    # print("delegated_puzzle: ", delegated_puzzle)

    print("NEW SOLUTION DATA ==============")
    # print("new_solution: ", new_solution)
    # print(list(new_solution.to_program().as_python()))
    new_lineage_proof, new_my_amount, new_inner_solution = new_solution.to_program().as_python()
    parent_parent_coin_info, parent_inner_puzzle_hash, parent_amount = new_lineage_proof
    from clvm.casts import int_from_bytes
    print("parent_amount: ", int_from_bytes(parent_amount))
    print("parent_inner_puzzle_hash: ", parent_inner_puzzle_hash.hex())
    print("parent_parent_coin_info: ", parent_parent_coin_info.hex())
    print("new_my_amount: ", int_from_bytes(new_my_amount))
    _, delegated_puzzle, _ = new_inner_solution
    # print("delegated_puzzle: ", delegated_puzzle)
    # ffffa030d940e53ed5b56fee3ae46ba5f4e59da5e2cc9242f6e482fe1f1e4d9a463639
    # ffa0c7b89cfb9abf2c4cb212a4840b37d762f4c880b8517b0dadb0c310ded24dd86dff82053980ff820539ffff80ffff01ffff31ffb0
    # c00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ffa09dcf97a1
    # 84f32623d11a73124ceb99a5709b083721e878a16d78f596718ba7b280ffff33ffa0c7b89cfb9abf2c4cb212a4840b37d762f4c880b8
    # 517b0dadb0c310ded24dd86dff8205358080ff808080
    # ffffa0039759eda861cd44c0af6c9501300f66fe4f5de144b8ae4fc4e8da35701f38ac
    # ffa0c7b89cfb9abf2c4cb212a4840b37d762f4c880b8517b0dadb0c310ded24dd86dff82053980ff820539ffff80ffff01ffff31ffb0
    # c00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000ffa09dcf97a1
    # 84f32623d11a73124ceb99a5709b083721e878a16d78f596718ba7b280ffff33ffa0c7b89cfb9abf2c4cb212a4840b37d762f4c880b8
    # 517b0dadb0c310ded24dd86dff8205358080ff808080
    singleton_child = None
    patched_additions = []
    for addition in spend_data.additions:
        patched_addition = Coin(unspent_lineage_ids.coin_id, addition.puzzle_hash, addition.amount)
        patched_additions.append(patched_addition)
        if addition.puzzle_hash == spend_data.coin_spend.coin.puzzle_hash:
            # We found the next version of this singleton
            print("We found the next version of this singleton")
            singleton_child = patched_addition
    if singleton_child is None:
        raise ValueError("Could not find fast forward child singleton.")
    new_coin_spend = CoinSpend(new_coin, spend_data.coin_spend.puzzle_reveal, new_solution)
    # Keep track of this in order to chain the next ff
    print("Keep track of this in order to chain the next ff")
    fast_forward_spends[spend_data.coin_spend.coin.puzzle_hash] = UnspentLineageIds(
        coin_id=singleton_child.name(),
        coin_amount=singleton_child.amount,
        parent_id=singleton_child.parent_coin_info,
        parent_amount=unspent_lineage_ids.coin_amount,
        parent_parent_id=unspent_lineage_ids.parent_id,
    )
    print("Exiting perform_the_fast_forward")
    return new_coin_spend, patched_additions


@dataclasses.dataclass(frozen=True)
class EligibleCoinSpends:
    deduplication_spends: Dict[bytes32, DedupCoinSpend] = dataclasses.field(default_factory=dict)
    fast_forward_spends: Dict[bytes32, UnspentLineageIds] = dataclasses.field(default_factory=dict)

    def get_deduplication_info(
        self, *, bundle_coin_spends: Dict[bytes32, BundleCoinSpend], max_cost: int
    ) -> Tuple[List[CoinSpend], uint64, List[Coin]]:
        """
        Checks all coin spends of a mempool item for deduplication eligibility and
        provides the caller with the necessary information that allows it to perform
        identical spend aggregation on that mempool item if possible

        Args:
            bundle_coin_spends: the mempool item's coin spends data
            max_cost: the maximum limit when running for cost

        Returns:
            List[CoinSpend]: list of unique coin spends in this mempool item
            uint64: the cost we're saving by deduplicating eligible coins
            List[Coin]: list of unique additions in this mempool item

        Raises:
            ValueError to skip the mempool item we're currently in, if it's
            attempting to spend an eligible coin with a different solution than the
            one we're already deduplicating on.
        """
        cost_saving = 0
        unique_coin_spends: List[CoinSpend] = []
        unique_additions: List[Coin] = []
        new_dedup_spends: Dict[bytes32, DedupCoinSpend] = {}
        # See if this item has coin spends that are eligible for deduplication
        for coin_id, spend_data in bundle_coin_spends.items():
            if not spend_data.eligible_for_dedup:
                unique_coin_spends.append(spend_data.coin_spend)
                unique_additions.extend(spend_data.additions)
                continue
            # See if we processed an item with this coin before
            dedup_coin_spend = self.deduplication_spends.get(coin_id)
            if dedup_coin_spend is None:
                # We didn't process an item with this coin before. If we end up including
                # this item, add this pair to deduplication_spends
                new_dedup_spends[coin_id] = DedupCoinSpend(spend_data.coin_spend.solution, None)
                unique_coin_spends.append(spend_data.coin_spend)
                unique_additions.extend(spend_data.additions)
                continue
            # See if the solution was identical
            current_solution, duplicate_cost = dataclasses.astuple(dedup_coin_spend)
            if current_solution != spend_data.coin_spend.solution:
                # It wasn't, so let's skip this whole item because it's relying on
                # spending this coin with a different solution and that would
                # conflict with the coin spends that we're deduplicating already
                # NOTE: We can miss an opportunity to deduplicate on other solutions
                # even if they end up saving more cost, as we're going for the first
                # solution we see from the relatively highest FPC item, to avoid
                # severe performance and/or time-complexity impact
                raise ValueError("Solution is different from what we're deduplicating on")
            # Let's calculate the saved cost if we never did that before
            if duplicate_cost is None:
                # See first if this mempool item had this cost computed before
                # This can happen if this item didn't get included in the previous block
                spend_cost = spend_data.cost
                if spend_cost is None:
                    spend_cost = run_for_cost(
                        puzzle_reveal=spend_data.coin_spend.puzzle_reveal,
                        solution=spend_data.coin_spend.solution,
                        additions_count=len(spend_data.additions),
                        max_cost=max_cost,
                    )
                    # Update this mempool item's coin spends map
                    bundle_coin_spends[coin_id] = BundleCoinSpend(
                        coin_spend=spend_data.coin_spend,
                        eligible_for_dedup=spend_data.eligible_for_dedup,
                        eligible_for_fast_forward=spend_data.eligible_for_fast_forward,
                        additions=spend_data.additions,
                        cost=spend_cost,
                    )
                duplicate_cost = spend_cost
                # If we end up including this item, update this entry's cost
                new_dedup_spends[coin_id] = DedupCoinSpend(current_solution, duplicate_cost)
            cost_saving += duplicate_cost
        # Update the eligible coin spends data
        self.deduplication_spends.update(new_dedup_spends)
        return unique_coin_spends, uint64(cost_saving), unique_additions

    async def process_fast_forward_spends(
        self,
        *,
        mempool_item: InternalMempoolItem,
        get_unspent_lineage_ids_for_puzzle_hash: Callable[[bytes32], Awaitable[Optional[UnspentLineageIds]]],
        height: uint32,
        constants: ConsensusConstants,
    ) -> None:
        """
        Provides the caller with an in-place internal mempool item that has a
        proper state of fast forwarded coin spends and additions starting from
        the most recent unspent versions of the related singleton spends.

        Args:
            mempool_item: the internal mempool item to process
            get_unspent_lineage_ids_for_puzzle_hash: to lookup the most recent
                version of the singleton from the coin store
            constants: needed in order to refresh the mempool item if needed
            height: needed in order to refresh the mempool item if needed

        Raises:
            If a fast forward cannot proceed, to prevent potential double spends
        """
        new_coin_spends = []
        ff_bundle_coin_spends = {}
        for coin_id, spend_data in mempool_item.bundle_coin_spends.items():
            print("=======")
            print("processing ff for coin id: ", coin_id.hex(), " and amount: ", spend_data.coin_spend.coin.amount)
            if not spend_data.eligible_for_fast_forward:
                # Nothing to do for this spend, moving on
                print("Nothing to do for this spend, moving on")
                new_coin_spends.append(spend_data.coin_spend)
                continue
            # See if we processed a fast forward item with this puzzle hash before
            print("See if we processed a fast forward item with this puzzle hash before")
            unspent_lineage_ids = self.fast_forward_spends.get(spend_data.coin_spend.coin.puzzle_hash)
            if unspent_lineage_ids is None:
                # We didn't, so let's lookup the most recent version from the DB
                print("We didn't, so let's lookup the most recent version from the DB")
                unspent_lineage_ids = await get_unspent_lineage_ids_for_puzzle_hash(
                    spend_data.coin_spend.coin.puzzle_hash
                )
                if unspent_lineage_ids is None:
                    raise ValueError("Cannot proceed with singleton spend fast forward.")
                # See if we're the most recent version
                print("See if we're the most recent version")
                if unspent_lineage_ids.coin_id == coin_id:
                    # We are, so we don't need to fast forward, we just need to
                    # set the next version from our additions to chain ff spends
                    print("We are, so we don't need to fast forward")
                    set_next_singleton_version(
                        current_singleton=spend_data.coin_spend.coin,
                        singleton_additions=spend_data.additions,
                        fast_forward_spends=self.fast_forward_spends,
                    )
                    # Nothing more to do for this spend, moving on
                    print("Nothing more to do for this spend, moving on")
                    new_coin_spends.append(spend_data.coin_spend)
                    continue
                # We're not the most recent version, so let's fast forward
                print("We're not the most recent version, so let's fast forward")
                print("our id: ", coin_id.hex())
                print("our amount: ", spend_data.coin_spend.coin.amount)
                print("latest id: ", unspent_lineage_ids.coin_id.hex())
                print("latest amount: ", unspent_lineage_ids.coin_amount)
                print("latest parent: ", unspent_lineage_ids.parent_id.hex())
                print("latest parent amount: ", unspent_lineage_ids.parent_amount)
                new_coin_spend, patched_additions = perform_the_fast_forward(
                    unspent_lineage_ids=unspent_lineage_ids,
                    spend_data=spend_data,
                    fast_forward_spends=self.fast_forward_spends,
                )
                # Mark this coin for a coin spend data update
                print("Mark this coin for a coin spend data update")
                ff_bundle_coin_spends[coin_id] = BundleCoinSpend(
                    coin_spend=new_coin_spend,
                    eligible_for_dedup=spend_data.eligible_for_dedup,
                    eligible_for_fast_forward=spend_data.eligible_for_fast_forward,
                    additions=patched_additions,
                    cost=spend_data.cost,
                )
                # Update the list of coins spends that will make the new fast
                # forward spend bundle
                print("Update the list of coins spends that will make the new fast")
                new_coin_spends.append(new_coin_spend)
                # We're done here, moving on
                print("We're done here, moving on")
                continue
            # We processed this puzzle hash before, so build on that
            # See first if we're the most recent version
            print("We processed this puzzle hash before, so build on that")
            print("coin_id: ", coin_id.hex())
            print("unspent_lineage_ids.coin_id: ", unspent_lineage_ids.coin_id.hex())
            if unspent_lineage_ids.coin_id == coin_id:
                # We are, so we don't need to fast forward, we just need to
                # find the next version among our additions to chain ff spends
                print("we don't need to fast forward")
                set_next_singleton_version(
                    current_singleton=spend_data.coin_spend.coin,
                    singleton_additions=spend_data.additions,
                    fast_forward_spends=self.fast_forward_spends,
                )
                # Nothing more to do for this spend, moving on
                new_coin_spends.append(spend_data.coin_spend)
                continue
            # We're not the most recent version, so let's fast forward
            print("We're not the most recent version, so let's fast forward")
            new_coin_spend, patched_additions = perform_the_fast_forward(
                unspent_lineage_ids=unspent_lineage_ids,
                spend_data=spend_data,
                fast_forward_spends=self.fast_forward_spends,
            )
            # Mark this coin for a coin spend data update
            ff_bundle_coin_spends[coin_id] = BundleCoinSpend(
                coin_spend=new_coin_spend,
                eligible_for_dedup=spend_data.eligible_for_dedup,
                eligible_for_fast_forward=spend_data.eligible_for_fast_forward,
                additions=patched_additions,
                cost=spend_data.cost,
            )
            # Update the list of coins spends that make the new fast forward bundle
            new_coin_spends.append(new_coin_spend)
        if len(ff_bundle_coin_spends) == 0:
            # This item doesn't have any fast forward coins, nothing to do here
            return
        # Update the mempool item after validating the new spend bundle
        new_sb = SpendBundle(
            coin_spends=new_coin_spends, aggregated_signature=mempool_item.spend_bundle.aggregated_signature
        )
        # We need to run the new spend bundle to make sure it remains valid
        new_npc_result = get_name_puzzle_conditions(
            generator=simple_solution_generator(new_sb),
            max_cost=mempool_item.npc_result.cost,
            mempool_mode=True,
            height=height,
            constants=constants,
        )
        if new_npc_result.error is not None:
            print("spend validation failed with error: ", new_npc_result.error)
            raise ValueError("Singleton spend fast forward became invalid.")
        # Update bundle_coin_spends using the collected data
        mempool_item.bundle_coin_spends.update(ff_bundle_coin_spends)
        # Update the mempool item with the new spend bundle related data
        dataclasses.replace(mempool_item, spend_bundle=new_sb, npc_result=new_npc_result)
