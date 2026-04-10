Set sh = CreateObject("WScript.Shell")
sh.Run "cmd /c """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\AnySort.cmd""", 0, False
