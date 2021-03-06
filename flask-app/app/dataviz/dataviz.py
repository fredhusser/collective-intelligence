from bokeh.models import ColumnDataSource, HoverTool, OpenURL, TapTool
from bokeh.plotting import figure
import numpy as np

__author__ = 'husser'


def get_grid(data_by_node, attributes):
    """Create the ColumnDataSource instance as used by Bokeh
    for plotting the grid. It transforms the pandas dataframe
    into the appropriate structure.

    Todo: link the node id to a list of articles IDs for further
    bindings to URIs in the database

    Parameters:
    -----------

    data_by_node: pandas dataframe
        Contains the node data with the list of nodes linking to
        the cluster ID and the main topics, and article hits

    attributes: dictionary
        Contains the main attributes of the data modelling session.
    """
    n_clusters = attributes["n_clusters"]
    k_shape = attributes["kshape"]
    colors = ["#%02x%02x%02x" % (np.floor(250.*(float(n)/n_clusters)), 100, 100)
              for n in xrange(n_clusters)]
    colormap = {i: colors[i] for i in xrange(n_clusters)}
    source = ColumnDataSource(data_by_node)
    source.add([colormap[cl] for cl in source.data["cluster"]],'color')

    x_range = [str(x) for x in xrange(k_shape[0])]
    y_range = [str(x) for x in xrange(k_shape[1])]
    return source, x_range, y_range


def render_grid(source, x_range, y_range, TOOLS="hover,tap"):
    """Function to render a bokeh figure with a rectangular grid
    representing the SOM nodes, colored by cluster id and showing
    the number of articles that they match. By hovering on the nodes
    one display the top topics.

    Parameters
    ----------

    source: ColumnDataSource
        Data to be displayed, as generated by get_grid method

    x_range, yrange: lists
        List used to scale the nodes coordinates
    """
    p = figure(title="Self-organizing map features and cluster ID",
               tools=TOOLS)
    p.plot_width = 800
    p.toolbar_location = "left"

    # Draw the rectangular color grid
    p.rect("x", "y", 1, 1, source=source,
           fill_alpha=0.6, color="color")

    # Write some content in the boxes
    text_props = {
        "source": source,
        "angle": 0,
        "color": "black",
        "text_align": "left",
        "text_baseline": "middle"
    }
    p.text(x="x", y="y", text="hits",
           text_font_style="bold", text_font_size="8pts", **text_props)
    p.grid.grid_line_color = None

    # Create a hover tool for representing the node data
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [("Node_id","@node")]+[("Top Words no. %d\t" % (i + 1),
                                   "@word_%d" % (i + 1)) for i in range(3)]

    # Create a link to the node data
    url = "/som/node/@node"
    taptool = p.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    return p


### Methods for plotting ###
def plot_map(matrix, kshape, title="New Map", topology="array"):
    """Implementation of the plot of the U-Matrix corresponding to
    a Kohonen map.

    Parameters:
    -----------

    U_matrix: CSR-sparse matrix
        The precomputed U_matrix from the som package (n_nodes,1)

    kshape: tuple (x,y)
        Shape of the SOM as the matrix is flattened over nodes

    figure_id: string or int
        Use this to refer to a new figure. If none, create the fig
        from nothing

    axes: string
        Use the identifier for the subplot eg: '111', '113', '123'

    topology: string
        Can be "array" for a standard square topology or "hex"
    """
    # Instantiate a matplotlib figure
    s = figure(width=250, plot_height=250, title=title,
               x_range=[0, kshape[0]], y_range=[0, kshape[1]])

    # Reshape the U_matrix
    map = np.asarray(np.split(matrix, kshape[0], axis=0))
    dw, dh = kshape
    # Plotting the axes
    s.image(image=[map], x=[0], y=[0], dw=[dw], dh=[dh], palette="Spectral11")
    return s


