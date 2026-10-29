from transformers import AutoTokenizer
                          

def enviar_contexto(mensaje, etiqueta, tokenizer, system_prompt):
    MAX_MESSAGE_TOKENS = 256
    MAX_SEQ_LENGTH = 384

    # Limitar la longitud del mensaje.
    message_ids = tokenizer.encode(
        str(mensaje),
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_MESSAGE_TOKENS,
    )

    mensaje_recortado = tokenizer.decode(
        message_ids,
        skip_special_tokens=True,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Mensaje a clasificar:\n{mensaje_recortado}",
        },
    ]

    # Generar primero la plantilla como texto para evitar
    # diferencias en el tipo de retorno de apply_chat_template.
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    # Etiqueta objetivo y token de fin de turno.
    target_ids = tokenizer.encode(
        str(int(etiqueta)),
        add_special_tokens=False,
    ) + tokenizer.encode(
        "<|im_end|>",
        add_special_tokens=False,
    )

    input_ids = prompt_ids + target_ids

    if len(input_ids) > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Secuencia de {len(input_ids)} tokens; "
            f"supera MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}."
        )

    # La pérdida se calcula solamente sobre la respuesta.
    labels = [-100] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
    }
