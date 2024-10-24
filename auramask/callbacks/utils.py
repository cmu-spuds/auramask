import io


def get_model_summary(model):
    stream = io.StringIO()
    model.summary(
        print_fn=lambda x: stream.write(x + "\n"),
        expand_nested=True,
        show_trainable=True,
    )
    summary_string = stream.getvalue()
    stream.close()
    return summary_string
