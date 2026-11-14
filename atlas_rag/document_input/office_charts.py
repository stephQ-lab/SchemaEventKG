from ssf import SSF


def cached_values(source):
    points = source.xpath("./c:numRef/c:numCache/c:pt | ./c:numLit/c:pt")
    values = {point.idx: point.v.text or "" for point in points}
    formats = source.xpath("./c:numRef/c:numCache/c:formatCode/text() | ./c:numLit/c:formatCode/text()")
    return values, formats[0] if formats else ""


def display_values(labels, fmt, chart):
    if fmt in ("", "General", "@"):
        return labels
    date1904 = chart._chartSpace.date1904 is not None and chart._chartSpace.date1904.val
    formatter = SSF(date1904=bool(date1904), locale="zh-CN", errors="raise")
    return [formatter.format(fmt, float(value))
            if value not in ("", "#N/A", "#VALUE!", "#REF!") else value for value in labels]


def chart_tables(chart):
    result = []
    for plot_index, plot in enumerate(chart.plots, 1):
        if plot.series and plot.series[0]._element.xpath("./c:val"):
            categories = [str(category.label) for category in plot.categories]
            original = list(categories)
            formats = plot.series[0]._element.xpath("./c:cat/c:numRef/c:numCache/c:formatCode/text() | ./c:cat/c:numLit/c:formatCode/text()")
            category_format = formats[0] if formats else ""
            categories = display_values(categories, category_format, chart)
            if not categories:
                indexes = [index for series in plot.series for index in cached_values(series._element.val)[0]]
                categories = ["横轴未确认（数据序号%d）" % (index + 1) for index in range(max(indexes, default=-1) + 1)]
            series_data = []
            for series in plot.series:
                values, fmt = cached_values(series._element.val)
                displayed = display_values([values.get(index, "") for index in range(len(categories))], fmt, chart)
                series_data.append({"name": series.name, "values": displayed,
                                    "number_format": fmt, "cached_points": values})
            rows = [["横轴标签"] + [item["name"] + ("（原图数字格式：%s）" % item["number_format"]
                                                   if item["number_format"] not in ("", "General") else "") for item in series_data]]
            for index, category in enumerate(categories):
                rows.append([category] + [item["values"][index] for item in series_data])
            result.append({"rows": rows, "header_rows": 1, "chart_type": str(chart.chart_type), "plot_index": plot_index,
                           "native_series": series_data, "native_categories": original,
                           "category_number_format": category_format, "category_verified": bool(original)})
        else:
            for series in plot.series:
                x_values, x_format = cached_values(series._element.xVal)
                y_values, y_format = cached_values(series._element.yVal)
                indexes = sorted(set(x_values) | set(y_values))
                x_labels = display_values([x_values.get(index, "") for index in indexes], x_format, chart)
                y_labels = display_values([y_values.get(index, "") for index in indexes], y_format, chart)
                rows = [["横轴标签", series.name]] + [[x, y] for x, y in zip(x_labels, y_labels)]
                result.append({"rows": rows, "header_rows": 1, "chart_type": str(chart.chart_type), "plot_index": plot_index,
                               "native_series": [{"name": series.name, "x_values": x_values, "values": y_values,
                                                  "x_number_format": x_format, "number_format": y_format}]})
    return result
