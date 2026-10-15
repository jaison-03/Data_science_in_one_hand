"""Playable Tic-Tac-Toe against a Minimax game-playing agent."""
import math
import streamlit as st

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Minimax Game AI",page_icon="❎",layout="centered")
st.markdown("<style>.block-container{max-width:650px;padding-top:2rem}.stButton>button{height:70px;font-size:28px}</style>",unsafe_allow_html=True)
st.title("Minimax: Tic-Tac-Toe")
st.caption("You are X. The minimax opponent is O and searches the complete game tree.")
if "minimax_board" not in st.session_state: st.session_state.minimax_board=[""]*9
if "minimax_over" not in st.session_state: st.session_state.minimax_over=False
def winner(board):
    for a,b,c in ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)):
        if board[a] and board[a]==board[b]==board[c]:return board[a]
    return None
def minimax(board,player,alpha=-math.inf,beta=math.inf):
    result=winner(board)
    if result=="O":return 10,None
    if result=="X":return -10,None
    if all(board):return 0,None
    moves=[]
    for i,cell in enumerate(board):
        if cell:continue
        board[i]=player
        score,_=minimax(board,"X" if player=="O" else "O",alpha,beta)
        board[i]=""
        moves.append((score,i))
        if player=="O":
            alpha=max(alpha,score)
            if beta<=alpha:break
        else:
            beta=min(beta,score)
            if beta<=alpha:break
    return (max(moves) if player=="O" else min(moves))
board=st.session_state.minimax_board
if not st.session_state.minimax_over:
    cols=st.columns(3)
    for i in range(9):
        if cols[i%3].button(board[i] or "·",key=f"cell-{i}",disabled=bool(board[i])):
            board[i]="X"
            if not winner(board) and not all(board):
                _,move=minimax(board,"O")
                if move is not None:board[move]="O"
            if winner(board) or all(board):st.session_state.minimax_over=True
            st.rerun()
result=winner(board)
if result:st.success("You win!" if result=="X" else "Minimax wins.")
elif all(board):st.info("Draw.")
else:st.info("Your turn — choose a cell.")
if st.button("New game"):
    st.session_state.minimax_board=[""]*9;st.session_state.minimax_over=False;st.rerun()
