# Custom row-wise DataFrame printing
function print_row(df::DataFrame, n; header=nothing, fmt="%-8.3g")
    show(
        @view df[n, :];
        noheader=(n == 1) ? false : true,
        nosubheader=(n == 1) ? true : false,
        show_row_number=false,
        title="",
        header=header,
        tf=(n == 1) ? DataFrames.tf_simple : DataFrames.tf_borderless,
        formatters=PrettyTables.ft_printf(fmt),
        allcols=true
    )
    println() # Add linebreak
end