# One-shot: back up the DACL of the path, then write a DACL that adds a FullControl
# allow ACE for the signed-in user. Only the DACL is written; owner, group, SACL and
# file contents are untouched. Windows PowerShell 5.1 compatible.
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Path,
  [Parameter(Mandatory = $true)][string]$Out
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not ('Dsh.AclFix' -as [type])) {
  Add-Type -Namespace Dsh -Name AclFix -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)]
public struct SID_AND_ATTRIBUTES { public IntPtr Sid; public uint Attributes; }
[StructLayout(LayoutKind.Sequential)]
public struct TOKEN_MANDATORY_LABEL { public SID_AND_ATTRIBUTES Label; }
[DllImport("advapi32.dll", SetLastError = true)]
public static extern bool GetTokenInformation(IntPtr token, int infoClass, IntPtr info, uint length, out uint returned);
[DllImport("advapi32.dll", SetLastError = true)]
public static extern IntPtr GetSidSubAuthority(IntPtr sid, uint index);
[DllImport("advapi32.dll", SetLastError = true)]
public static extern IntPtr GetSidSubAuthorityCount(IntPtr sid);
[DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
private static extern Microsoft.Win32.SafeHandles.SafeFileHandle CreateFileW(
  string path, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
public static bool HasAccess(string path, uint access) {
  using (var h = CreateFileW(path, access, 7, IntPtr.Zero, 3, 0x02200000, IntPtr.Zero)) {
    if (!h.IsInvalid) { return true; }
    int err = Marshal.GetLastWin32Error();
    if (err == 5) { return false; }
    throw new System.ComponentModel.Win32Exception(err, "CreateFileW access check: " + path);
  }
}
[DllImport("advapi32.dll", CharSet = CharSet.Unicode)]
private static extern uint SetNamedSecurityInfoW(string path, int type, uint information,
  IntPtr owner, IntPtr group, byte[] dacl, IntPtr sacl);
public static void SetDacl(string path, string sddl) {
  var d = new System.Security.AccessControl.RawSecurityDescriptor(sddl);
  byte[] dacl = null;
  if (d.DiscretionaryAcl != null) {
    dacl = new byte[d.DiscretionaryAcl.BinaryLength];
    d.DiscretionaryAcl.GetBinaryForm(dacl, 0);
  }
  bool prot = (d.ControlFlags & System.Security.AccessControl.ControlFlags.DiscretionaryAclProtected) != 0;
  uint r = SetNamedSecurityInfoW(path, 1, 4u | (prot ? 0x80000000u : 0x20000000u),
    IntPtr.Zero, IntPtr.Zero, dacl, IntPtr.Zero);
  if (r != 0) { throw new System.ComponentModel.Win32Exception((int)r, "Write DACL: " + path); }
}
public static int Level(IntPtr token) {
  uint length;
  GetTokenInformation(token, 25, IntPtr.Zero, 0, out length);
  if (length == 0) { throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "len"); }
  IntPtr buf = Marshal.AllocHGlobal((int)length);
  try {
    if (!GetTokenInformation(token, 25, buf, length, out length)) { throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), "info"); }
    IntPtr sid = Marshal.PtrToStructure<TOKEN_MANDATORY_LABEL>(buf).Label.Sid;
    byte count = Marshal.ReadByte(GetSidSubAuthorityCount(sid));
    return Marshal.ReadInt32(GetSidSubAuthority(sid, (uint)(count - 1)));
  } finally { Marshal.FreeHGlobal(buf); }
}
'@
}

$full = [System.IO.Path]::GetFullPath($Path)
$outFull = [System.IO.Path]::GetFullPath($Out)
$me = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$meSid = $me.User.Value

if ($full.Length -gt 3) { $full = $full.TrimEnd('\') }
Write-Output "TARGET $full"
Write-Output "CALLER_SID $meSid"
try {
  $lvl = [Dsh.AclFix]::Level($me.Token)
  Write-Output "INTEGRITY S-1-16-$lvl"
} catch { Write-Output "INTEGRITY unknown: $($_.Exception.Message)" }

if (-not [System.IO.Directory]::Exists($full)) { throw "not an existing directory: $full" }
[System.IO.Directory]::CreateDirectory($outFull) | Out-Null

Write-Output ("ACCESS_BEFORE writeDac={0} writeOwner={1}" -f ([Dsh.AclFix]::HasAccess($full, 0x40000)), ([Dsh.AclFix]::HasAccess($full, 0x80000)))

$acl = Get-Acl -LiteralPath $full
$observed = $acl.GetSecurityDescriptorSddlForm([System.Security.AccessControl.AccessControlSections]::Access)

# Keep the object's own ACEs and its protection flags; inherited ACEs follow the parent.
$flags = [regex]::Match($observed, '^D:([A-Z]*)').Groups[1].Value
$aces = @([regex]::Matches($observed, '\([^()]*\)') | ForEach-Object { $_.Value } | Where-Object { ($_ -split ';')[1] -notmatch 'ID' })
$ownDacl = 'D:{0}{1}' -f $flags, ($aces -join '')

$record = Join-Path $outFull ("acl-backup-{0}.json" -f [guid]::NewGuid().ToString('N'))
@{
  Path       = $full
  Dacl       = $ownDacl
  Protected  = $acl.AreAccessRulesProtected
  Observed   = $observed
  TimeUtc    = [datetime]::UtcNow.ToString('o')
} | ConvertTo-Json | Set-Content -LiteralPath $record -Encoding utf8
Write-Output "BACKUP_FILE $record"
Write-Output "OLD_OWN_DACL $ownDacl"

$mineBefore = @($acl.GetAccessRules($true, $false, [System.Security.Principal.SecurityIdentifier]) |
  Where-Object { $_.IdentityReference.Value -eq $meSid -and $_.AccessControlType -eq 'Allow' })
Write-Output ("BEFORE_MY_ALLOW_ACES {0}" -f $mineBefore.Count)

$acl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new(
  [System.Security.Principal.SecurityIdentifier]::new($meSid),
  [System.Security.AccessControl.FileSystemRights]::FullControl,
  [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [System.Security.AccessControl.InheritanceFlags]::ObjectInherit,
  [System.Security.AccessControl.PropagationFlags]::None,
  [System.Security.AccessControl.AccessControlType]::Allow))

$sddlToWrite = $acl.GetSecurityDescriptorSddlForm([System.Security.AccessControl.AccessControlSections]::Access)
Write-Output "WRITING_OWN_DACL $sddlToWrite"
[Dsh.AclFix]::SetDacl($full, $sddlToWrite)

$after = Get-Acl -LiteralPath $full
$mineAfter = @($after.GetAccessRules($true, $false, [System.Security.Principal.SecurityIdentifier]) |
  Where-Object { $_.IdentityReference.Value -eq $meSid -and $_.AccessControlType -eq 'Allow' })
Write-Output ("AFTER_MY_ALLOW_ACES {0}" -f $mineAfter.Count)
foreach ($rule in $mineAfter) {
  Write-Output ("  ACE rights={0} type={1} inherited={2} inherit={3}" -f $rule.FileSystemRights, $rule.AccessControlType, $rule.IsInherited, $rule.InheritanceFlags)
}
Write-Output ("ACCESS_AFTER writeDac={0} writeOwner={1}" -f ([Dsh.AclFix]::HasAccess($full, 0x40000)), ([Dsh.AclFix]::HasAccess($full, 0x80000)))
Write-Output "DONE"
