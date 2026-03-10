def group_original_id_tokens(original_id: object) -> list[dict[str, object]]:
    """Split an original_id into characters and mark pair gaps within digit runs."""
    if original_id is None:
        return []

    id_str = str(original_id)
    tokens: list[dict[str, object]] = []
    digit_run_index = 0

    for index, char in enumerate(id_str):
        if char.isdigit():
            digit_run_index += 1
            next_char_is_digit = index + 1 < len(id_str) and id_str[index + 1].isdigit()
            gap_after = next_char_is_digit and digit_run_index % 2 == 0
        else:
            digit_run_index = 0
            gap_after = False

        tokens.append({"char": char, "gap_after": gap_after})

    return tokens
