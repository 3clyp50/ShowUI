def batch_add_answer(source, answer):
    """Add answer to the last assistant message in the conversation."""
    if isinstance(source, list):
        # Find the last assistant message
        for i in range(len(source) - 1, -1, -1):
            if source[i]["role"] == "assistant":
                source[i]["content"] = answer
                break
    return source

def batch_add_answer_append(source, answer):
    """Append answer to the last assistant message in the conversation."""
    if isinstance(source, list):
        # Find the last assistant message
        for i in range(len(source) - 1, -1, -1):
            if source[i]["role"] == "assistant":
                if isinstance(source[i]["content"], str):
                    source[i]["content"] += answer
                else:
                    # If content is a list of dicts, append to the last text content
                    for j in range(len(source[i]["content"]) - 1, -1, -1):
                        if "text" in source[i]["content"][j]:
                            source[i]["content"][j]["text"] += answer
                            break
                break
    return source
