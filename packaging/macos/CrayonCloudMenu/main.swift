// Crayon Cloud menu bar helper (macOS). The .app's main executable. On first launch it builds a Python environment
// under ~/Library/Application Support/CrayonCloud (from the package source shipped in Contents/Resources), then it
// starts `crayoncloud serve`, shows a menu bar item (status, Open, Copy address for ButterKnife, network switch,
// log, Quit), and stops the server when quitting. Built by tools/make-macos-app.sh with swiftc; no Xcode project.
import AppKit
import Foundation

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var statusLine: NSMenuItem!
    private var openItem: NSMenuItem!
    private var copyItem: NSMenuItem!
    private var lanItem: NSMenuItem!
    private var assetsItem: NSMenuItem!
    private var assetsUrl: URL?
    private var addressLine: NSMenuItem!
    private var server: Process?
    private var localUrl = URL(string: "http://127.0.0.1:8765/")!
    private var lanUrl: URL?
    private var quitting = false
    private var restarting = false
    private var log: FileHandle?
    private var buffer = Data()
    private var signalSources: [DispatchSourceSignal] = []
    private var healthTimer: Timer?

    private let port = 8765
    private let support = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/CrayonCloud")
    private var venv: URL { support.appendingPathComponent("venv") }
    private var version: String { Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0" }
    private var lanEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: "lan") }
        set { UserDefaults.standard.set(newValue, forKey: "lan") }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Crayon Cloud exists to serve ButterKnife, which is often on another machine: reachable on the network by default.
        UserDefaults.standard.register(defaults: ["lan": true])
        buildMenu()
        openLog()
        installSignalHandlers()
        setStatus("Starting…")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            do {
                try self.ensureEnvironment()
                DispatchQueue.main.async { self.startServer() }
            } catch {
                DispatchQueue.main.async { self.fail("Crayon Cloud could not set itself up", error.localizedDescription) }
            }
        }
    }

    private func installSignalHandlers() {
        for sig in [SIGTERM, SIGINT, SIGHUP] {
            signal(sig, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler { NSApp.terminate(nil) }
            source.resume()
            signalSources.append(source)
        }
    }

    private func buildMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let image = Bundle.main.image(forResource: "MenuIcon") {
            image.isTemplate = true
            statusItem.button?.image = image
        } else {
            statusItem.button?.title = "CC"
        }
        statusItem.button?.toolTip = "Crayon Cloud"

        let menu = NSMenu()
        statusLine = NSMenuItem(title: "Starting…", action: nil, keyEquivalent: "")
        statusLine.isEnabled = false
        openItem = NSMenuItem(title: "Open Crayon Cloud", action: #selector(openBrowser), keyEquivalent: "o")
        openItem.target = self
        openItem.isEnabled = false
        copyItem = NSMenuItem(title: "Copy address for ButterKnife", action: #selector(copyAddress), keyEquivalent: "c")
        copyItem.target = self
        copyItem.isEnabled = false
        addressLine = NSMenuItem(title: "", action: nil, keyEquivalent: "")
        addressLine.isEnabled = false
        addressLine.isHidden = true
        assetsItem = NSMenuItem(title: "Open Assets Folder", action: #selector(openAssets), keyEquivalent: "a")
        assetsItem.target = self
        assetsItem.isEnabled = false
        lanItem = NSMenuItem(title: "Reachable on the local network", action: #selector(toggleLan), keyEquivalent: "")
        lanItem.target = self
        lanItem.state = lanEnabled ? .on : .off
        let logItem = NSMenuItem(title: "Show log", action: #selector(showLog), keyEquivalent: "l")
        logItem.target = self
        let quitItem = NSMenuItem(title: "Quit Crayon Cloud", action: #selector(quit), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(statusLine)
        menu.addItem(addressLine)
        menu.addItem(.separator())
        menu.addItem(openItem)
        menu.addItem(copyItem)
        menu.addItem(assetsItem)
        menu.addItem(lanItem)
        menu.addItem(.separator())
        menu.addItem(logItem)
        menu.addItem(quitItem)
        menu.autoenablesItems = false
        statusItem.menu = menu
    }

    private func setStatus(_ text: String) {
        statusLine.title = text
        statusItem.button?.toolTip = "Crayon Cloud: \(text)"
    }

    // MARK: - Environment (first run)

    /// The interpreter shipped inside the bundle (python-build-standalone, put there by tools/make-macos-app.sh).
    private var bundledPython: URL? {
        let url = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/python/bin/python3")
        return FileManager.default.isExecutableFile(atPath: url.path) ? url : nil
    }

    /// The bundled Python, else a 3.10–3.13 on this machine: Homebrew, python.org, MacPorts or the system one.
    private func findPython() -> URL? {
        if let bundled = bundledPython { return bundled }
        var candidates: [String] = []
        for minor in [12, 13, 11, 10] {
            candidates += ["/opt/homebrew/bin/python3.\(minor)", "/usr/local/bin/python3.\(minor)",
                           "/Library/Frameworks/Python.framework/Versions/3.\(minor)/bin/python3.\(minor)", "/opt/local/bin/python3.\(minor)"]
        }
        candidates += ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            if let out = try? run(path, ["-c", "import sys; print(sys.version_info.minor)"]), let minor = Int(out.trimmingCharacters(in: .whitespacesAndNewlines)), (10...13).contains(minor) {
                return URL(fileURLWithPath: path)
            }
        }
        return nil
    }

    /// Creates the virtual environment and installs the shipped package into it; reinstalls when the app version changed.
    private func ensureEnvironment() throws {
        let marker = venv.appendingPathComponent(".crayoncloud-version")
        let installed = (try? String(contentsOf: marker, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
        let command = venv.appendingPathComponent("bin/crayoncloud")
        let venvPython = venv.appendingPathComponent("bin/python").path
        // A venv points at the interpreter that made it. If that Python is gone (the app moved, or a machine Python
        // was removed after the bundled one arrived), the environment is dead and is rebuilt from scratch.
        let healthy = FileManager.default.isExecutableFile(atPath: venvPython) && ((try? run(venvPython, ["-c", "import sys"])) != nil) && ((try? runStatus(venvPython, ["-c", "import sys"])) == 0)
        if installed == version && healthy && FileManager.default.isExecutableFile(atPath: command.path) { return }
        if FileManager.default.fileExists(atPath: venvPython) && !healthy {
            writeLog("=== the environment's Python is gone; rebuilding\n")
            try? FileManager.default.removeItem(at: venv)
        }

        guard let source = Bundle.main.url(forResource: "crayoncloud-src", withExtension: nil) else {
            throw NSError(domain: "CrayonCloud", code: 1, userInfo: [NSLocalizedDescriptionKey: "The app bundle has no crayoncloud-src folder; rebuild it with tools/make-macos-app.sh."])
        }
        guard let python = findPython() else {
            throw NSError(domain: "CrayonCloud", code: 2, userInfo: [NSLocalizedDescriptionKey: "No Python 3.10 to 3.13 was found. Install one with Homebrew (brew install python@3.12) or from python.org, then open Crayon Cloud again."])
        }

        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        DispatchQueue.main.async { self.setStatus(installed == nil ? "Setting up (first run, a few minutes)…" : "Updating to \(self.version)…") }
        writeLog("=== Crayon Cloud \(version): setting up \(venv.path) with \(python.path)\(bundledPython == nil ? "" : " (bundled)")\n")
        if !FileManager.default.fileExists(atPath: venv.appendingPathComponent("bin/python").path) {
            try runLogged(python.path, ["-m", "venv", venv.path])
        }
        let pip = venv.appendingPathComponent("bin/python").path
        try runLogged(pip, ["-m", "pip", "install", "--upgrade", "pip"])
        try runLogged(pip, ["-m", "pip", "install", "--upgrade", "\(source.path)[mlx]"])
        try version.write(to: marker, atomically: true, encoding: .utf8)
        writeLog("=== ready\n")
    }

    private func runStatus(_ path: String, _ arguments: [String]) throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    }

    private func run(_ path: String, _ arguments: [String]) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return String(decoding: data, as: UTF8.self)
    }

    private func runLogged(_ path: String, _ arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if !data.isEmpty { self?.log?.write(data) }
        }
        try process.run()
        process.waitUntilExit()
        pipe.fileHandleForReading.readabilityHandler = nil
        if process.terminationStatus != 0 {
            throw NSError(domain: "CrayonCloud", code: Int(process.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "\(URL(fileURLWithPath: path).lastPathComponent) \(arguments.prefix(3).joined(separator: " ")) failed (exit \(process.terminationStatus)). See the log: ~/Library/Logs/CrayonCloud/server.log"])
        }
    }

    // MARK: - Server

    private func openLog() {
        let logs = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/CrayonCloud")
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let file = logs.appendingPathComponent("server.log")
        if !FileManager.default.fileExists(atPath: file.path) {
            FileManager.default.createFile(atPath: file.path, contents: nil)
        }
        log = try? FileHandle(forWritingTo: file)
        log?.seekToEndOfFile()
    }

    private func writeLog(_ text: String) { log?.write(Data(text.utf8)) }

    private func startServer() {
        let process = Process()
        process.executableURL = venv.appendingPathComponent("bin/crayoncloud")
        var arguments = ["serve", "--port", String(port), "--parent-pid", String(ProcessInfo.processInfo.processIdentifier)]
        if lanEnabled { arguments.append("--lan") }
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { return }
            DispatchQueue.main.async { self?.consume(data) }
        }
        process.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                guard let self = self else { return }
                if self.restarting {
                    self.restarting = false
                    self.startServer()
                } else if !self.quitting {
                    self.fail("Crayon Cloud stopped", "The server exited unexpectedly. The log may say why: ~/Library/Logs/CrayonCloud/server.log")
                }
            }
        }

        do {
            setStatus("Starting server…")
            openItem.isEnabled = false
            copyItem.isEnabled = false
            addressLine.isHidden = true
            lanUrl = nil
            try process.run()
            server = process
            healthTimer?.invalidate()
            healthTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in self?.pollHealth() }
        } catch {
            fail("Crayon Cloud could not start", error.localizedDescription)
        }
    }

    private func consume(_ data: Data) {
        log?.write(data)
        buffer.append(data)
        while let newline = buffer.firstIndex(of: 0x0A) {
            let line = String(decoding: buffer[buffer.startIndex..<newline], as: UTF8.self)
            buffer.removeSubrange(buffer.startIndex...newline)
            parse(line)
        }
    }

    private func parse(_ line: String) {
        guard let range = line.range(of: " is running at ") else { return }
        var rest = line[range.upperBound...]
        if let space = rest.firstIndex(of: " ") { rest = rest[rest.startIndex..<space] }
        if let url = URL(string: String(rest)) { localUrl = url }
        if let lanRange = line.range(of: "(on your network: "), let end = line[lanRange.upperBound...].firstIndex(of: ")") {
            lanUrl = URL(string: String(line[lanRange.upperBound..<end]))
        }
        openItem.isEnabled = true
        copyItem.isEnabled = true
        if let lan = lanUrl {
            addressLine.title = "On your network: \(lan.absoluteString)v1"
            copyItem.title = "Copy network address for ButterKnife"
        } else {
            addressLine.title = "This Mac only: \(localUrl.absoluteString)v1"
            copyItem.title = "Copy address for ButterKnife"
        }
        addressLine.isHidden = false
        setStatus("Running")
    }

    /// /health says whether the model is loaded and busy; the status line follows it.
    private func pollHealth() {
        guard server?.isRunning == true, openItem.isEnabled else { return }
        var request = URLRequest(url: localUrl.appendingPathComponent("health"))
        request.timeoutInterval = 1.5
        URLSession.shared.dataTask(with: request) { [weak self] data, _, _ in
            guard let self = self, let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            let loaded = json["loaded"] as? Bool ?? false
            let busy = json["busy"] as? Bool ?? false
            let seconds = json["seconds_busy"] as? Double
            let model = json["model"] as? String ?? "model"
            let assets = json["assets"] as? String
            let text = busy ? "Rendering… \(Int(seconds ?? 0)) s" : loaded ? "Ready (\(model) loaded)" : "Ready (\(model) loads on the first picture)"
            DispatchQueue.main.async {
                self.setStatus(text)
                if let assets = assets {
                    self.assetsUrl = URL(fileURLWithPath: assets)
                    self.assetsItem.isEnabled = true
                }
            }
        }.resume()
    }

    private func fail(_ title: String, _ message: String) {
        setStatus(title)
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "Show log")
        alert.addButton(withTitle: "Quit")
        NSApp.activate(ignoringOtherApps: true)
        if alert.runModal() == .alertFirstButtonReturn { showLog() }
        NSApp.terminate(nil)
    }

    // MARK: - Menu actions

    @objc private func openBrowser() { NSWorkspace.shared.open(localUrl) }

    @objc private func copyAddress() {
        let base = (lanUrl ?? localUrl).absoluteString
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(base.hasSuffix("/") ? base + "v1" : base + "/v1", forType: .string)
    }

    /// The folder the server keeps rendered pictures in (~/Pictures/Crayon Cloud unless told otherwise); created if empty.
    @objc private func openAssets() {
        guard let url = assetsUrl else { return }
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.open(url)
    }

    @objc private func toggleLan() {
        lanEnabled.toggle()
        lanItem.state = lanEnabled ? .on : .off
        guard let server = server, server.isRunning else { return }
        restarting = true
        setStatus("Restarting…")
        server.terminate()
    }

    @objc private func showLog() {
        let file = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/CrayonCloud/server.log")
        NSWorkspace.shared.open(file)
    }

    @objc private func quit() { NSApp.terminate(nil) }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        quitting = true
        healthTimer?.invalidate()
        guard let server = server, server.isRunning else { return .terminateNow }
        server.terminate()
        let deadline = Date().addingTimeInterval(5)
        while server.isRunning && Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.1))
        }
        if server.isRunning { kill(server.processIdentifier, SIGKILL) }
        return .terminateNow
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
