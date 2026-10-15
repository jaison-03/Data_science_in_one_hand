"""Interactive grid A* pathfinding visualizer with editable obstacles."""
import heapq
import random
import plotly.graph_objects as go
import streamlit as st

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="A* Pathfinding",page_icon="🧭",layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("A* Pathfinding Visualizer")
st.caption("Edit obstacle coordinates as row,column pairs. A* uses 4-way movement and Manhattan distance.")
rows=st.slider("Grid rows",8,30,16)
cols=st.slider("Grid columns",8,30,22)
default="3,5 3,6 3,7 4,7 5,7 6,7 7,7 8,7 9,7 10,7 11,7 11,8 11,9"
wall_text=st.text_input("Obstacle cells (row,column separated by spaces)",st.session_state.get("astar_walls",default))
try:
    walls={tuple(map(int,item.split(","))) for item in wall_text.split() if item.strip()}
    if any(len(cell)!=2 or not(0<=cell[0]<rows and 0<=cell[1]<cols) for cell in walls): raise ValueError("Obstacle coordinates must fit the grid.")
except ValueError as error:
    st.error(f"Invalid obstacle coordinates: {error}");st.stop()
start=(0,0);goal=(rows-1,cols-1)
if start in walls or goal in walls: st.error("Start and goal cells cannot be obstacles.");st.stop()
def astar(start,goal,obstacles,height,width):
    queue=[(0,0,start)]; came={}; cost={start:0}; visited=set();sequence=0
    while queue:
        _,_,current=heapq.heappop(queue)
        if current in visited: continue
        visited.add(current)
        if current==goal: break
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nxt=(current[0]+dr,current[1]+dc)
            if not(0<=nxt[0]<height and 0<=nxt[1]<width) or nxt in obstacles: continue
            new_cost=cost[current]+1
            if new_cost<cost.get(nxt,float("inf")):
                cost[nxt]=new_cost;came[nxt]=current;sequence+=1
                heuristic=abs(goal[0]-nxt[0])+abs(goal[1]-nxt[1])
                heapq.heappush(queue,(new_cost+heuristic,sequence,nxt))
    if goal not in cost:return [],visited
    path=[goal]
    while path[-1]!=start:path.append(came[path[-1]])
    return path[::-1],visited
path,visited=astar(start,goal,walls,rows,cols)
z=[[0 for _ in range(cols)] for _ in range(rows)]
for r,c in walls:z[r][c]=1
for r,c in visited:z[r][c]=2
for r,c in path:z[r][c]=3
z[start[0]][start[1]]=4;z[goal[0]][goal[1]]=5
fig=go.Figure(go.Heatmap(z=z,zmin=0,zmax=5,colorscale=[[0,"#f8fafc"],[.2,"#111827"],[.4,"#bfdbfe"],[.6,"#22c55e"],[.8,"#3b82f6"],[1,"#f59e0b"]],showscale=False))
fig.update_xaxes(side="top",dtick=1);fig.update_yaxes(autorange="reversed",dtick=1,scaleanchor="x")
fig.update_layout(height=650,title="Start (amber) · Goal (gold) · Path (green) · Visited (blue) · Walls (dark)")
st.plotly_chart(fig,use_container_width=True)
if path: st.success(f"Optimal path length: {len(path)-1} steps · visited cells: {len(visited)}")
else: st.error(f"No path found · visited cells: {len(visited)}")
with st.expander("Randomize obstacles"):
    density=st.slider("Wall density",.05,.4,.18,.01)
    if st.button("Generate random grid"):
        rng=random.Random(8)
        walls={(r,c) for r in range(rows) for c in range(cols) if rng.random()<density and (r,c) not in (start,goal)}
        st.session_state["astar_walls"] = " ".join(f"{r},{c}" for r,c in sorted(walls))
        st.rerun()
