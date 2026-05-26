#!/usr/bin/env python3
"""
ABB IO signal test script — read or write any IO signal via RWS.

Usage:
  # Read current value
  python test_io_signal.py --ip 192.168.125.1 --signal DoValve1

  # Set signal ON
  python test_io_signal.py --ip 192.168.125.1 --signal DoValve1 --value 1

  # Set signal OFF with explicit network/unit
  python test_io_signal.py --ip 192.168.125.1 --signal sig1 --network Virtual --unit DUNIT --value 0

  # List all signals on the controller
  python test_io_signal.py --ip 192.168.125.1 --list
"""

import argparse
import sys

DEFAULT_IP       = "192.168.125.1"
DEFAULT_USERNAME = "Default User"
DEFAULT_PASSWORD = "robotics"


def _rws_get(ip: str, username: str, password: str, path: str) -> dict:
    """GET a RWS JSON endpoint, returning the parsed response."""
    import requests
    from requests.auth import HTTPDigestAuth
    url = f"http://{ip}/{path.lstrip('/')}"
    resp = requests.get(url, auth=HTTPDigestAuth(username, password), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _resolve_signal_path(ip: str, username: str, password: str, signal: str) -> tuple[str, str]:
    """Return (network, unit) for a signal by querying RWS.
    The _title field in the response is 'Network/Unit/SignalName'."""
    try:
        data = _rws_get(ip, username, password, f"/rw/iosystem/signals/{signal}?json=1")
        title = data["_embedded"]["_state"][0].get("_title", "")
        parts = title.split("/")
        if len(parts) == 3:
            return parts[0], parts[1]
    except Exception:
        pass
    return "Local", "DRV_1"


def _connect(ip: str, username: str, password: str):
    try:
        from abb_robot_client.rws import RWS
    except ImportError:
        print("ERROR: abb_robot_client not installed — pip install abb-robot-client")
        sys.exit(1)

    base_url = f"http://{ip}"
    print(f"Connecting to {base_url} (user={username!r}) …", flush=True)
    rws = RWS(base_url=base_url, username=username, password=password)
    try:
        state = rws.get_controller_state()
        print(f"Connected.  controller_state={state!r}\n", flush=True)
    except Exception as e:
        print(f"ERROR: connection failed — {e}")
        print("  Check: IP reachable? RWS enabled? Correct credentials?")
        sys.exit(1)
    return rws


def _list_signals(ip: str, username: str, password: str) -> None:
    """List all IO signals via the raw RWS iosystem endpoint."""
    print(f"Fetching signal list from http://{ip}/rw/iosystem/signals …\n", flush=True)
    try:
        data = _rws_get(ip, username, password, "/rw/iosystem/signals?json=1")
    except Exception as e:
        print(f"ERROR fetching signal list: {e}")
        sys.exit(1)

    try:
        signals = data["_embedded"]["_state"]
    except (KeyError, TypeError):
        import json
        print("Unexpected response format:")
        print(json.dumps(data, indent=2))
        return

    if not signals:
        print("(no signals found)")
        return

    col = "{:<30} {:<20} {:<15} {:<8} {}"
    print(col.format("Signal", "Network/Unit", "Unit", "Type", "Value"))
    print("-" * 80)
    for sig in signals:
        # _title is "Network/Unit/SignalName" — parse it for the real path
        title  = sig.get("_title", "")
        parts  = title.split("/")
        network = parts[0] if len(parts) == 3 else "?"
        unit    = parts[1] if len(parts) == 3 else sig.get("unitnm", "?")
        print(col.format(
            sig.get("name",   "?"),
            f"{network}/{unit}",
            unit,
            sig.get("type",   "?"),
            sig.get("lvalue", "?"),
        ))


def _read_signal(rws, signal: str, network: str, unit: str) -> int:
    try:
        return rws.get_digital_io(signal, network=network, unit=unit)
    except Exception as e:
        print(f"ERROR reading '{signal}': {e}")
        sys.exit(1)


def _set_signal(rws, signal: str, value: int, network: str, unit: str) -> None:
    mastered = False
    try:
        rws.request_mastership("iosystem")
        mastered = True
    except AttributeError:
        pass  # older library version without explicit mastership
    except Exception as e:
        print(f"WARNING: request_mastership(iosystem) failed (ignored): {e}", flush=True)

    try:
        rws.set_digital_io(signal, value, network=network, unit=unit)
    except Exception as e:
        print(f"ERROR setting '{signal}' → {value}: {e}")
        sys.exit(1)
    finally:
        if mastered:
            try:
                rws.release_mastership("iosystem")
            except Exception:
                pass


def main() -> None:
    p = argparse.ArgumentParser(
        description="Read/write ABB robot IO signals via RWS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ip",       default=DEFAULT_IP,
                   help="Controller IP address (default: %(default)s)")
    p.add_argument("--username", default=DEFAULT_USERNAME,
                   help="RWS username (default: %(default)r)")
    p.add_argument("--password", default=DEFAULT_PASSWORD,
                   help="RWS password (default: %(default)r)")
    p.add_argument("--signal",
                   help="IO signal name, e.g. DoValve1 or DO_VACUUM")
    p.add_argument("--network",  default=None,
                   help="Signal network (auto-detected from RWS if omitted)")
    p.add_argument("--unit",     default=None,
                   help="Signal unit (auto-detected from RWS if omitted)")
    p.add_argument("--value",    type=int, choices=[0, 1],
                   help="Value to write: 0=off, 1=on  (omit to read only)")
    p.add_argument("--list",     action="store_true",
                   help="List all IO signals on the controller")
    args = p.parse_args()

    if args.list:
        _list_signals(args.ip, args.username, args.password)
        return

    if not args.signal:
        p.error("--signal is required (or use --list to browse available signals)")

    # Auto-detect network/unit from RWS if not given explicitly
    if args.network is None or args.unit is None:
        network, unit = _resolve_signal_path(args.ip, args.username, args.password, args.signal)
        network = args.network or network
        unit    = args.unit    or unit
    else:
        network, unit = args.network, args.unit

    rws = _connect(args.ip, args.username, args.password)

    current = _read_signal(rws, args.signal, network, unit)
    print(f"Signal  : {args.signal}")
    print(f"Network : {network}   Unit: {unit}")
    print(f"Current : {current}  ({'ON' if current else 'OFF'})")

    if args.value is None:
        print("\n(No --value specified — read-only mode)")
        return

    print(f"\nSetting {args.signal} → {args.value} ({'ON' if args.value else 'OFF'}) …",
          flush=True)
    _set_signal(rws, args.signal, args.value, network, unit)

    new_val = _read_signal(rws, args.signal, network, unit)
    ok = new_val == args.value
    print(f"Readback: {new_val}  ({'ON' if new_val else 'OFF'})  [{'OK' if ok else 'MISMATCH'}]")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
