import plotly.graph_objects as go
from plotly.subplots import make_subplots

labels = ["US", "China", "European Union", "Russian Federation", "Brazil", "India",
          "Rest of World"]
labels2 = ["USA", "China", "European Union", "Russian Federation", "Brazil", "India",
          "Rest of World"]
fig = make_subplots(rows=2, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=labels, values=[16, 15, 12, 6, 5, 4, 42], name="IPC Degradation"),
              row = 1, col = 1)
fig.add_trace(go.Pie(labels=labels2, values=[27, 11, 25, 8, 1, 3, 25], name="IPC Degradation"),
              row = 1, col = 2)
fig.add_trace(go.Pie(labels=labels, values=[16, 15, 12, 6, 5, 4, 42], name="IPC Degradation"),
              row = 2, col = 1)
fig.add_trace(go.Pie(labels=labels, values=[27, 11, 25, 8, 1, 3, 25], name="CO2 Emissions"),
              row = 2, col = 2)

fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
            title_text="Diagrfam with results")
                # Add annotations in the center of the donut pies.
                    #annotations=[dict(text='GHG', x=0.18, y=0.5, font_size=20, showarrow=False),
                                         #dict(text='CO2', x=0.82, y=0.5, font_size=20, showarrow=False)])
fig.show()
