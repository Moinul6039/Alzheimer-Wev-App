const { useState, useEffect } = React;

const STAGE_STYLES = {
    "Non Demented": "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
    "Very Mild Demented": "bg-amber-500/15 text-amber-300 ring-amber-500/30",
    "Mild Demented": "bg-orange-500/15 text-orange-300 ring-orange-500/30",
    "Moderate Demented": "bg-rose-500/15 text-rose-300 ring-rose-500/30",
};

function stageBadgeClass(stage) {
    return STAGE_STYLES[stage] || "bg-sky-500/15 text-sky-300 ring-sky-500/30";
}

function parseConfidence(value) {
    if (!value || value === "N/A") return 0;
    const parsed = parseFloat(String(value).replace("%", ""));
    return Number.isFinite(parsed) ? Math.min(100, Math.max(0, parsed)) : 0;
}

function Spinner() {
    return (
        <svg
            className="h-5 w-5 animate-spin text-slate-900"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            aria-hidden="true"
        >
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
        </svg>
    );
}

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewSrc, setPreviewSrc] = useState("");
    const [previewText, setPreviewText] = useState("No file selected.");
    const [statusMessage, setStatusMessage] = useState("");
    const [statusType, setStatusType] = useState("info");
    const [prediction, setPrediction] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const [dragActive, setDragActive] = useState(false);

    useEffect(() => {
        const stored = window.localStorage.getItem("alzheimerHistory");
        if (stored) {
            setHistory(JSON.parse(stored));
        }
    }, []);

    const storeHistory = (entry) => {
        setHistory((currentHistory) => {
            const nextHistory = [entry, ...currentHistory].slice(0, 10);
            window.localStorage.setItem("alzheimerHistory", JSON.stringify(nextHistory));
            return nextHistory;
        });
    };

    const handleClearHistory = () => {
        setHistory([]);
        window.localStorage.removeItem("alzheimerHistory");
    };

    const handleSelectedFile = (file) => {
        if (!file || !file.type.startsWith("image/")) {
            setStatusMessage("Please upload a valid image file (JPG or PNG).");
            setStatusType("error");
            return;
        }

        setSelectedFile(file);
        setPreviewText(file.name);
        setPrediction(null);
        setConfidence(null);
        setStatusMessage("");
        setStatusType("info");

        const reader = new FileReader();
        reader.onload = () => setPreviewSrc(reader.result);
        reader.readAsDataURL(file);
    };

    const handleFileChange = (event) => {
        handleSelectedFile(event.target.files[0]);
    };

    const handleDrop = (event) => {
        event.preventDefault();
        setDragActive(false);
        handleSelectedFile(event.dataTransfer.files[0]);
    };

    const handlePredict = async () => {
        if (!selectedFile) {
            setStatusMessage("Choose an image before predicting.");
            setStatusType("error");
            return;
        }

        setIsLoading(true);
        setStatusMessage("Analyzing image...");
        setStatusType("info");
        setPrediction(null);
        setConfidence(null);

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Prediction failed");
            }

            const data = await response.json();
            const resultPrediction = data.prediction || "Unknown";
            const resultConfidence = data.confidence ? `${data.confidence}%` : "N/A";

            setPrediction(resultPrediction);
            setConfidence(resultConfidence);
            setStatusMessage("Prediction complete.");
            setStatusType("success");

            storeHistory({
                id: Date.now(),
                name: selectedFile.name,
                prediction: resultPrediction,
                confidence: resultConfidence,
                date: new Date().toLocaleString(),
            });
        } catch (error) {
            setStatusMessage(`Error: ${error.message}`);
            setStatusType("error");
        } finally {
            setIsLoading(false);
        }
    };

    const handleReset = () => {
        setSelectedFile(null);
        setPreviewSrc("");
        setPreviewText("No file selected.");
        setPrediction(null);
        setConfidence(null);
        setStatusMessage("");
        setStatusType("info");
    };

    const confidenceValue = parseConfidence(confidence);
    const statusClasses =
        statusType === "error"
            ? "text-rose-300"
            : statusType === "success"
              ? "text-emerald-300"
              : "text-slate-400";

    const cardClass =
        "rounded-2xl border border-slate-800/80 bg-slate-900/70 p-6 shadow-xl shadow-black/20 backdrop-blur-sm";

    return (
        <div className="relative min-h-screen overflow-hidden">
            <div
                className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_15%_10%,rgba(56,189,248,0.14),transparent_22%),radial-gradient(circle_at_90%_15%,rgba(56,189,248,0.1),transparent_18%),radial-gradient(circle_at_60%_90%,rgba(59,130,246,0.08),transparent_25%)]"
                aria-hidden="true"
            />

            <main className="relative mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8">
                <header className="mb-10 text-center">
                    <p className="mb-3 inline-flex items-center gap-2 rounded-full bg-sky-500/10 px-4 py-1.5 text-xs font-semibold uppercase tracking-widest text-sky-400 ring-1 ring-sky-500/20">
                        <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />
                        Alzheimer Detection
                    </p>
                    <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl lg:text-6xl">
                        Brain scan analysis made simple
                    </h1>
                    <p className="mx-auto mt-4 max-w-2xl text-base leading-relaxed text-slate-400 sm:text-lg">
                        Upload an MRI scan, run the model, and view the predicted dementia stage with confidence
                        scores. History is saved locally in your browser.
                    </p>
                </header>

                <div className="grid gap-6 lg:grid-cols-5 lg:gap-8">
                    <section
                        className={`lg:col-span-3 ${cardClass} transition-all duration-200 ${
                            dragActive
                                ? "border-sky-500/60 bg-sky-500/5 ring-2 ring-sky-500/30"
                                : ""
                        }`}
                        onDragOver={(event) => {
                            event.preventDefault();
                            setDragActive(true);
                        }}
                        onDragLeave={() => setDragActive(false)}
                        onDrop={handleDrop}
                    >
                        <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
                            <div>
                                <h2 className="text-xl font-semibold text-white">Upload your scan</h2>
                                <p className="mt-1 text-sm text-slate-400">
                                    Drop a JPG or PNG here, or browse from your computer.
                                </p>
                            </div>
                            <span className="rounded-full bg-sky-500/10 px-3 py-1 text-xs font-bold text-sky-400 ring-1 ring-sky-500/25">
                                ResNet-18
                            </span>
                        </div>

                        <label
                            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 transition-all duration-200 ${
                                dragActive
                                    ? "border-sky-400 bg-sky-500/10"
                                    : "border-slate-700 bg-slate-800/40 hover:border-sky-500/50 hover:bg-slate-800/70"
                            }`}
                        >
                            <svg
                                className="mb-3 h-10 w-10 text-slate-500"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                                strokeWidth="1.5"
                                aria-hidden="true"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                                />
                            </svg>
                            <span className="text-sm font-medium text-slate-300">
                                Select image or drag it here
                            </span>
                            <span className="mt-1 text-xs text-slate-500">JPG, PNG — max one file at a time</span>
                            <input
                                id="imageInput"
                                type="file"
                                accept="image/*"
                                className="sr-only"
                                onChange={handleFileChange}
                            />
                        </label>

                        <div className="mt-6 flex min-h-[280px] flex-col items-center justify-center rounded-xl border border-slate-800/60 bg-slate-950/50 p-4">
                            {previewSrc ? (
                                <img
                                    src={previewSrc}
                                    alt="Selected scan preview"
                                    className="max-h-72 w-full rounded-lg object-contain"
                                />
                            ) : (
                                <div className="flex flex-col items-center text-slate-500">
                                    <svg
                                        className="mb-2 h-12 w-12 opacity-40"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                        strokeWidth="1"
                                        aria-hidden="true"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z"
                                        />
                                    </svg>
                                    <p className="text-sm">Preview will appear here</p>
                                </div>
                            )}
                            <p className="mt-3 max-w-full truncate text-sm text-slate-400">{previewText}</p>
                        </div>

                        <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                            <button
                                type="button"
                                onClick={handlePredict}
                                disabled={!selectedFile || isLoading}
                                className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-sky-400 to-cyan-400 px-5 py-3 text-sm font-semibold text-slate-900 transition hover:from-sky-300 hover:to-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                {isLoading ? (
                                    <>
                                        <Spinner />
                                        Analyzing...
                                    </>
                                ) : (
                                    "Run Prediction"
                                )}
                            </button>
                            <button
                                type="button"
                                onClick={handleReset}
                                disabled={isLoading}
                                className="rounded-xl border border-slate-700 bg-slate-800/60 px-5 py-3 text-sm font-semibold text-slate-200 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
                            >
                                Reset
                            </button>
                        </div>

                        {statusMessage ? (
                            <p className={`mt-4 text-sm ${statusClasses}`} role="status">
                                {statusMessage}
                            </p>
                        ) : null}
                    </section>

                    <aside className="flex flex-col gap-6 lg:col-span-2">
                        <section className={cardClass}>
                            <h2 className="text-lg font-semibold text-white">Prediction result</h2>
                            <p className="mt-1 text-sm text-slate-400">Latest run from the model.</p>

                            {prediction ? (
                                <div className="mt-5 space-y-4">
                                    <div>
                                        <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-500">
                                            Stage
                                        </p>
                                        <span
                                            className={`inline-flex rounded-full px-3 py-1 text-sm font-semibold ring-1 ring-inset ${stageBadgeClass(prediction)}`}
                                        >
                                            {prediction}
                                        </span>
                                    </div>
                                    <div>
                                        <div className="mb-2 flex items-center justify-between text-sm">
                                            <span className="font-medium text-slate-400">Confidence</span>
                                            <span className="font-semibold text-white">{confidence}</span>
                                        </div>
                                        <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                                            <div
                                                className="h-full rounded-full bg-gradient-to-r from-sky-400 to-cyan-400 transition-all duration-500"
                                                style={{ width: `${confidenceValue}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <p className="mt-5 text-sm leading-relaxed text-slate-500">
                                    No prediction yet. Upload a scan and click Run Prediction.
                                </p>
                            )}
                        </section>

                        <section className={`${cardClass} flex flex-1 flex-col`}>
                            <div className="mb-4 flex items-start justify-between gap-3">
                                <div>
                                    <h2 className="text-lg font-semibold text-white">Prediction history</h2>
                                    <p className="mt-1 text-sm text-slate-400">Last 10 runs, stored in this browser.</p>
                                </div>
                                {history.length ? (
                                    <button
                                        type="button"
                                        onClick={handleClearHistory}
                                        className="shrink-0 rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-300 transition hover:bg-slate-800"
                                    >
                                        Clear
                                    </button>
                                ) : null}
                            </div>

                            {history.length ? (
                                <ul className="max-h-80 space-y-2 overflow-y-auto pr-1">
                                    {history.map((item) => (
                                        <li
                                            key={item.id}
                                            className="flex items-start justify-between gap-3 rounded-xl border border-slate-800/80 bg-slate-800/40 p-3"
                                        >
                                            <div className="min-w-0 flex-1">
                                                <p className="truncate text-sm font-medium text-slate-200">
                                                    {item.name}
                                                </p>
                                                <p className="mt-0.5 text-xs text-slate-500">{item.date}</p>
                                            </div>
                                            <div className="shrink-0 text-right">
                                                <span
                                                    className={`inline-block rounded-md px-2 py-0.5 text-xs font-semibold ring-1 ring-inset ${stageBadgeClass(item.prediction)}`}
                                                >
                                                    {item.prediction}
                                                </span>
                                                <p className="mt-1 text-xs text-slate-400">{item.confidence}</p>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <p className="text-sm leading-relaxed text-slate-500">
                                    No predictions yet. Run one to see it here.
                                </p>
                            )}
                        </section>
                    </aside>
                </div>

                <p className="mt-10 text-center text-xs text-slate-600">
                    For research and educational use only — not a substitute for professional medical diagnosis.
                </p>
            </main>
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
