MODULE abb_tcp_server
!
! ABB IRC5 TCP command server — companion to app/robots/abb_tcp.py
!
! Load this module into T_ROB1 and run it.  It listens on TCP_PORT,
! accepts one client at a time, and executes motion/IO commands synchronously.
! Responses are only sent after the command completes, so the Python driver
! knows motion is done when it receives "OK".
!
! Protocol (line-oriented ASCII, newline-terminated):
!   PING                              → OK PONG
!   GETPOSE                           → OK x y z rx ry rz    (mm, deg ZYX)
!   GETJOINTS                         → OK j1 j2 j3 j4 j5 j6 (deg)
!   MOVEL    x y z rx ry rz vtcp vori → OK  (after MoveL completes)
!   MOVEJ    x y z rx ry rz vtcp vori → OK  (after MoveJ completes)
!   MOVEABSJ j1 j2 j3 j4 j5 j6 v vo  → OK  (after MoveAbsJ completes)
!   SETDO    signal_name 0|1          → OK
!   STOP                              → OK
!   (anything else)                   → ERR UNKNOWN:<verb>
!
! Configuration — edit the two constants below if needed:
CONST num TCP_PORT := 10100;
CONST string ACTIVE_TOOL := "tool0";
CONST string ACTIVE_WOBJ := "wobj0";

VAR socketdev srv_socket;
VAR socketdev cli_socket;
VAR bool      client_alive;
VAR string    recv_line;

! ── Entry point ──────────────────────────────────────────────────────────────
PROC main()
    SocketCreate srv_socket;
    SocketBind   srv_socket, "0.0.0.0", TCP_PORT;
    SocketListen srv_socket;
    TPWrite "TCPSrv: listening on port " \Num:=TCP_PORT;

    WHILE TRUE DO
        SocketAccept srv_socket, cli_socket \Time:=WAIT_MAX;
        TPWrite "TCPSrv: client connected";
        client_alive := TRUE;

        WHILE client_alive DO
            SocketReceive cli_socket \Str:=recv_line \Time:=WAIT_MAX;
            handle_cmd trim_line(recv_line);
        ENDWHILE

        SocketClose cli_socket;
        TPWrite "TCPSrv: client disconnected";
    ENDWHILE

    ERROR
        IF ERRNO = ERR_SOCK_CLOSED THEN
            client_alive := FALSE;
            TRYNEXT;
        ENDIF
        SocketClose cli_socket;
        client_alive := FALSE;
        RETRY;
ENDPROC

! ── Dispatch ─────────────────────────────────────────────────────────────────
PROC handle_cmd(string cmd)
    VAR string verb;
    verb := word_n(cmd, 1);

    IF     verb = "PING"     THEN send_ok "PONG";
    ELSEIF verb = "GETPOSE"  THEN handle_getpose;
    ELSEIF verb = "GETJOINTS" THEN handle_getjoints;
    ELSEIF verb = "MOVEL"    THEN handle_movel    cmd;
    ELSEIF verb = "MOVEJ"    THEN handle_movej    cmd;
    ELSEIF verb = "MOVEABSJ" THEN handle_moveabsj cmd;
    ELSEIF verb = "SETDO"    THEN handle_setdo    cmd;
    ELSEIF verb = "STOP"     THEN
        StopMove;
        send_ok "";
    ELSEIF StrLen(verb) = 0 THEN
        ! empty line — ignore
    ELSE
        send_err "UNKNOWN:" + verb;
    ENDIF
ENDPROC

! ── Command handlers ─────────────────────────────────────────────────────────
PROC handle_getpose()
    VAR robtarget rt;
    VAR num rx;
    VAR num ry;
    VAR num rz;
    rt := CRobT(\Tool:=tool0 \WObj:=wobj0);
    rx := EulerZYX(\X, rt.rot);
    ry := EulerZYX(\Y, rt.rot);
    rz := EulerZYX(\Z, rt.rot);
    send_ok   NumToStr(rt.trans.x, 3) + " "
            + NumToStr(rt.trans.y, 3) + " "
            + NumToStr(rt.trans.z, 3) + " "
            + NumToStr(rx, 4) + " "
            + NumToStr(ry, 4) + " "
            + NumToStr(rz, 4);
    ERROR
        send_err "GETPOSE_FAILED";
ENDPROC

PROC handle_getjoints()
    VAR jointtarget jt;
    jt := CJointT();
    send_ok   NumToStr(jt.robax.rax_1, 4) + " "
            + NumToStr(jt.robax.rax_2, 4) + " "
            + NumToStr(jt.robax.rax_3, 4) + " "
            + NumToStr(jt.robax.rax_4, 4) + " "
            + NumToStr(jt.robax.rax_5, 4) + " "
            + NumToStr(jt.robax.rax_6, 4);
    ERROR
        send_err "GETJOINTS_FAILED";
ENDPROC

PROC handle_movel(string cmd)
    VAR num x;
    VAR num y;
    VAR num z;
    VAR num rx;
    VAR num ry;
    VAR num rz;
    VAR num vtcp;
    VAR num vori;
    VAR robtarget tgt;
    VAR speeddata  spd;

    x    := numword(cmd, 2);
    y    := numword(cmd, 3);
    z    := numword(cmd, 4);
    rx   := numword(cmd, 5);
    ry   := numword(cmd, 6);
    rz   := numword(cmd, 7);
    vtcp := numword(cmd, 8);
    vori := numword(cmd, 9);
    IF vtcp <= 0 THEN vtcp := 200; ENDIF
    IF vori <= 0 THEN vori :=  50; ENDIF

    tgt := [[x, y, z], OrientZYX(rz, ry, rx),
             CRobT(\Tool:=tool0 \WObj:=wobj0).robconf,
             [0, 0, 0, 0, 0, 0]];
    spd := [vtcp, vori, 5000, 1000];

    MoveL tgt, spd, fine, tool0 \WObj:=wobj0;
    send_ok "";
    ERROR
        send_err "MOVEL_FAILED:" + NumToStr(ERRNO, 0);
ENDPROC

PROC handle_movej(string cmd)
    VAR num x;
    VAR num y;
    VAR num z;
    VAR num rx;
    VAR num ry;
    VAR num rz;
    VAR num vtcp;
    VAR num vori;
    VAR robtarget tgt;
    VAR speeddata  spd;

    x    := numword(cmd, 2);
    y    := numword(cmd, 3);
    z    := numword(cmd, 4);
    rx   := numword(cmd, 5);
    ry   := numword(cmd, 6);
    rz   := numword(cmd, 7);
    vtcp := numword(cmd, 8);
    vori := numword(cmd, 9);
    IF vtcp <= 0 THEN vtcp := 200; ENDIF
    IF vori <= 0 THEN vori :=  50; ENDIF

    tgt := [[x, y, z], OrientZYX(rz, ry, rx),
             CRobT(\Tool:=tool0 \WObj:=wobj0).robconf,
             [0, 0, 0, 0, 0, 0]];
    spd := [vtcp, vori, 5000, 1000];

    MoveJ tgt, spd, fine, tool0 \WObj:=wobj0;
    send_ok "";
    ERROR
        send_err "MOVEJ_FAILED:" + NumToStr(ERRNO, 0);
ENDPROC

PROC handle_moveabsj(string cmd)
    VAR num j1;
    VAR num j2;
    VAR num j3;
    VAR num j4;
    VAR num j5;
    VAR num j6;
    VAR num vtcp;
    VAR num vori;
    VAR jointtarget jt;
    VAR speeddata   spd;

    j1   := numword(cmd, 2);
    j2   := numword(cmd, 3);
    j3   := numword(cmd, 4);
    j4   := numword(cmd, 5);
    j5   := numword(cmd, 6);
    j6   := numword(cmd, 7);
    vtcp := numword(cmd, 8);
    vori := numword(cmd, 9);
    IF vtcp <= 0 THEN vtcp := 200; ENDIF
    IF vori <= 0 THEN vori :=  50; ENDIF

    jt  := [[j1, j2, j3, j4, j5, j6], [0, 0, 0, 0, 0, 0]];
    spd := [vtcp, vori, 5000, 1000];

    MoveAbsJ jt, spd, fine, tool0 \WObj:=wobj0;
    send_ok "";
    ERROR
        send_err "MOVEABSJ_FAILED:" + NumToStr(ERRNO, 0);
ENDPROC

PROC handle_setdo(string cmd)
    VAR string    signame;
    VAR num       val;
    VAR signaldo  the_do;

    signame := word_n(cmd, 2);
    val     := numword(cmd, 3);

    AliasIO signame, the_do;
    IF val >= 1 THEN
        SetDO the_do, 1;
    ELSE
        SetDO the_do, 0;
    ENDIF
    send_ok "";
    ERROR
        send_err "SETDO_FAILED:" + signame;
ENDPROC

! ── Socket helpers ────────────────────────────────────────────────────────────
PROC send_line(string s)
    SocketSend cli_socket \Str:=(s + "\0A");
    ERROR
        IF ERRNO = ERR_SOCK_CLOSED THEN
            client_alive := FALSE;
            TRYNEXT;
        ENDIF
ENDPROC

PROC send_ok(string payload)
    IF StrLen(payload) > 0 THEN
        send_line "OK " + payload;
    ELSE
        send_line "OK";
    ENDIF
ENDPROC

PROC send_err(string msg)
    send_line "ERR " + msg;
ENDPROC

! ── String parsing helpers ────────────────────────────────────────────────────

! Return the n-th space-separated word (1-indexed).
FUNC string word_n(string s, num n)
    VAR num i;
    VAR num start;
    VAR num count;
    VAR num len;
    len   := StrLen(s);
    count := 0;
    i     := 1;
    WHILE i <= len DO
        ! skip spaces
        WHILE i <= len AND StrPart(s, i, 1) = " " DO
            i := i + 1;
        ENDWHILE
        IF i > len THEN RETURN ""; ENDIF
        ! found a word
        start := i;
        count := count + 1;
        WHILE i <= len AND StrPart(s, i, 1) <> " " DO
            i := i + 1;
        ENDWHILE
        IF count = n THEN
            RETURN StrPart(s, start, i - start);
        ENDIF
    ENDWHILE
    RETURN "";
ENDFUNC

! Return the n-th word as a number (0 if missing or invalid).
FUNC num numword(string s, num n)
    VAR num  val;
    VAR bool ok;
    ok := StrToVal(word_n(s, n), val);
    IF NOT ok THEN val := 0; ENDIF
    RETURN val;
ENDFUNC

! Strip trailing CR / LF from a received line.
FUNC string trim_line(string s)
    VAR num len;
    len := StrLen(s);
    WHILE len > 0 AND
          (StrPart(s, len, 1) = "\0D" OR StrPart(s, len, 1) = "\0A") DO
        len := len - 1;
    ENDWHILE
    IF len = 0 THEN RETURN ""; ENDIF
    RETURN StrPart(s, 1, len);
ENDFUNC

ENDMODULE
