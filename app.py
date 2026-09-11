TypeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/heatseeker/app.py", line 264, in <module>
    render_heatmap(matrix, result["spot"], f"{result['ticker']} — GEX by strike & expiration")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/heatseeker/app.py", line 122, in render_heatmap
    fig.add_hline(
    ~~~~~~~~~~~~~^
        y=y_labels[closest_idx],
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        annotation_position="right",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.14/site-packages/plotly/graph_objs/_figure.py", line 524, in add_hline
    return super().add_hline(
           ~~~~~~~~~~~~~~~~~^
        y, row, col, exclude_empty_subplots, annotation, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.14/site-packages/plotly/basedatatypes.py", line 4084, in add_hline
    self._process_multiple_axis_spanning_shapes(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        dict(
        ^^^^^
    ...<11 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.14/site-packages/plotly/basedatatypes.py", line 4003, in _process_multiple_axis_spanning_shapes
    augmented_annotation = shapeannotation.axis_spanning_shape_annotation(
        annotation, shape_type, shape_args, annotation_kwargs
    )
File "/home/adminuser/venv/lib/python3.14/site-packages/plotly/shapeannotation.py", line 258, in axis_spanning_shape_annotation
    shape_dict = annotation_params_for_line(
        shape_type, shape_args, annotation_position
    )
File "/home/adminuser/venv/lib/python3.14/site-packages/plotly/shapeannotation.py", line 100, in annotation_params_for_line
    eY = _mean(Y)
File "/home/adminuser/venv/lib/python3.14/site-packages/plotly/shapeannotation.py", line 35, in _mean
    return float(sum(x)) / len(x)
                 ~~~^^^
