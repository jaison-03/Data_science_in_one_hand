"""Sudoku and map-colouring constraint satisfaction solver."""
import streamlit as st

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Constraint Satisfaction Solver",page_icon="🧩",layout="wide")
st.markdown("<style>.block-container{max-width:1050px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Constraint Satisfaction Problems")
mode=st.radio("Problem",["Sudoku","Map coloring"],horizontal=True)

def solve_sudoku(board):
    steps=[]
    def candidates(row,col):
        used=set(board[row])
        used.update(board[r][col] for r in range(9))
        br,bc=3*(row//3),3*(col//3)
        used.update(board[r][c] for r in range(br,br+3) for c in range(bc,bc+3))
        return [n for n in range(1,10) if n not in used]
    def search():
        best=None;choices=None
        for r in range(9):
            for c in range(9):
                if board[r][c]==0:
                    available=candidates(r,c)
                    if not available:return False
                    if choices is None or len(available)<len(choices):
                        best=(r,c);choices=available
                        if len(choices)==1:break
            if choices is not None and len(choices)==1:break
        if best is None:return True
        r,c=best
        for value in choices:
            board[r][c]=value;steps.append((r,c,value))
            if search():return True
            board[r][c]=0;steps.append((r,c,0))
        return False
    return search(),steps

if mode=="Sudoku":
    st.caption("Enter 9 rows of 9 digits. Use 0 or . for blank cells.")
    puzzle=st.text_area("Puzzle",value="530070000\n600195000\n098000060\n800060003\n400803001\n700020006\n060000280\n000419005\n000080079",height=190)
    if st.button("Solve Sudoku",type="primary"):
        rows=[line.strip().replace(".","0").replace(" ","") for line in puzzle.splitlines() if line.strip()]
        try:
            if len(rows)!=9 or any(len(row)!=9 or not row.isdigit() for row in rows):raise ValueError("Provide exactly 9 rows of 9 digits (0 or . for blank).")
            board=[[int(value) for value in row] for row in rows]
            if any(value<0 or value>9 for row in board for value in row):raise ValueError("Sudoku values must be between 0 and 9.")
            for r in range(9):
                values=[board[r][c] for c in range(9) if board[r][c]]
                if len(values)!=len(set(values)):raise ValueError(f"Conflicting givens in row {r+1}.")
                values=[board[c][r] for c in range(9) if board[c][r]]
                if len(values)!=len(set(values)):raise ValueError(f"Conflicting givens in column {r+1}.")
            for br in range(0,9,3):
                for bc in range(0,9,3):
                    values=[board[r][c] for r in range(br,br+3) for c in range(bc,bc+3) if board[r][c]]
                    if len(values)!=len(set(values)):raise ValueError("Puzzle has duplicate values in a 3×3 box.")
            solved,steps=solve_sudoku(board)
            if not solved:st.error("No solution exists for this puzzle.")
            else:
                st.success(f"Solved with {len(steps)} assignment/backtrack steps.")
                st.dataframe(board,hide_index=True,use_container_width=True)
                reveal=st.slider("Show solver progress step",0,len(steps),len(steps))
                preview=[[int(value) for value in row] for row in rows]
                for r,c,value in steps[:reveal]:preview[r][c]=value
                st.subheader("Step-by-step board")
                st.dataframe(preview,hide_index=True,use_container_width=True)
        except ValueError as error:st.error(str(error))
else:
    adjacency={"North":["West","Central"],"West":["North","Central","South"],"Central":["North","West","South","East"],"South":["West","Central","East"],"East":["Central","South"]}
    colors=st.multiselect("Available colors",["Red","Green","Blue","Yellow"],default=["Red","Green","Blue"])
    assignments={}
    def color_search():
        if len(assignments)==len(adjacency):return True
        region=max((r for r in adjacency if r not in assignments),key=lambda r:sum(n in assignments for n in adjacency[r]))
        for color in colors:
            if all(assignments.get(neighbor)!=color for neighbor in adjacency[region]):
                assignments[region]=color
                if color_search():return True
                assignments.pop(region)
        return False
    if st.button("Solve map coloring",type="primary"):
        if not colors:st.error("Choose at least one color.")
        elif color_search():
            st.success("A valid coloring was found.")
            st.dataframe([{"Region":region,"Color":color} for region,color in assignments.items()],hide_index=True)
        else:st.error("No valid coloring exists with the selected palette.")
    with st.expander("Adjacency constraints"):st.json(adjacency)
