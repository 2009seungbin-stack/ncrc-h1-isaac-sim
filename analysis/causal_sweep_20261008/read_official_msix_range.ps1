param([long]$Start = -1, [long]$Length = 0, [long]$ArchiveLength = 0)
$ErrorActionPreference = 'Stop'
$uri = 'https://persistent.oaistatic.com/codex-app-prod/ChatGPT-x64.msix'
$client = [System.Net.Http.HttpClient]::new()
$request = [System.Net.Http.HttpRequestMessage]::new()
$request.RequestUri = [uri]$uri
$request.Method = if ($Start -lt 0) { [System.Net.Http.HttpMethod]::Head } else { [System.Net.Http.HttpMethod]::Get }
if ($Start -ge 0) {
    if ($Length -le 0 -or $Length -gt 8388608) { throw 'Metadata request outside size limit' }
    $rangeEnd = $Start + $Length - 1
    $request.Headers.Range = [System.Net.Http.Headers.RangeHeaderValue]::new($Start, $rangeEnd)
}
try {
    $response = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
    $response.EnsureSuccessStatusCode() | Out-Null
    if ($Start -lt 0) {
        @{length=$response.Content.Headers.ContentLength; modified=[string]$response.Content.Headers.LastModified} | ConvertTo-Json -Compress
    } else {
        $contentRange = [string]$response.Content.Headers.ContentRange
        $expectedRange = "bytes $Start-$rangeEnd/$ArchiveLength"
        if ([int]$response.StatusCode -ne 206 -or $contentRange -ne $expectedRange) { throw 'Server did not honor exact byte range' }
        $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
        if ($bytes.LongLength -ne $Length) { throw 'Range length mismatch' }
        @{body=[Convert]::ToBase64String($bytes)} | ConvertTo-Json -Compress
    }
} finally {
    if ($response) { $response.Dispose() }
    $request.Dispose()
    $client.Dispose()
}
