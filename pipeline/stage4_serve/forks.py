"""The 4 pre-curated Fork Explorer decision points."""

from pipeline.common.schemas import ForkPoint

FORK_POINTS: dict[str, ForkPoint] = {
    fp.id: fp
    for fp in [
        ForkPoint(
            id="slytherin-sorting",
            title="What if Harry was Sorted into Slytherin?",
            setup_summary=(
                "The Sorting Hat hesitates over Harry's head in his first year, "
                "weighing Slytherin against Gryffindor before he protests."
            ),
            fork_book=1,
            fork_chapter=7,
            character_pov="Harry Potter",
            system_prompt_override=(
                "Canon ends the moment the Sorting Hat is about to call out a "
                "House for Harry. From here, the Hat sorts him into Slytherin "
                "instead of Gryffindor. Continue the scene and its immediate "
                "aftermath in Harry's voice, showing how he and the people "
                "around him react to this divergence."
            ),
        ),
        ForkPoint(
            id="forest-no-return",
            title="What if Harry let Voldemort kill him in the forest and didn't come back?",
            setup_summary=(
                "Harry walks into the Forbidden Forest to face Voldemort, "
                "believing this is the only way to end the war."
            ),
            fork_book=7,
            fork_chapter=34,
            character_pov="Harry Potter",
            system_prompt_override=(
                "Canon ends the instant the Killing Curse hits Harry in the "
                "Forbidden Forest. From here, Harry does not return -- there is "
                "no King's Cross, no second chance. Continue the scene from "
                "Harry's perspective as it diverges from canon, exploring what "
                "this choice means for him and for the war."
            ),
        ),
        ForkPoint(
            id="hermione-stays-behind",
            title="What if Hermione chose to stay behind instead of following Harry and Ron?",
            setup_summary=(
                "As the trio scatters after the attack on the Burrow during "
                "Bill and Fleur's wedding, Hermione faces the choice of "
                "whether to go on the run with Harry and Ron."
            ),
            fork_book=7,
            fork_chapter=9,
            character_pov="Hermione Granger",
            system_prompt_override=(
                "Canon ends right after the wedding is attacked and the trio "
                "is about to Disapparate together. From here, Hermione decides "
                "to stay behind instead of joining Harry and Ron on the hunt "
                "for Horcruxes. Continue the scene in Hermione's voice, "
                "showing her reasoning and what she does next."
            ),
        ),
        ForkPoint(
            id="draco-accepts-offer",
            title="What if Draco accepted Dumbledore's offer at the top of the Astronomy Tower?",
            setup_summary=(
                "Draco corners Dumbledore at the top of the Astronomy Tower, "
                "wand raised, and Dumbledore offers him protection in exchange "
                "for lowering his wand."
            ),
            fork_book=6,
            fork_chapter=27,
            character_pov="Draco Malfoy",
            system_prompt_override=(
                "Canon ends the moment Dumbledore offers to protect Draco and "
                "his family if he puts his wand down. From here, Draco accepts "
                "the offer instead of the Death Eaters arriving. Continue the "
                "scene from Draco's perspective as it diverges from canon."
            ),
        ),
    ]
}
