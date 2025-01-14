const { exec } = require('child_process');

let script = process.argv[3];
let result = {
    "ram": { "start": process.resourceUsage().maxRSS * 1024 },
    "errors": {},
    "stdout": null,
    "stderr": null,
    "exit_code": null,
}

exec(`node ${script}`, (error, stdout, stderr) => {
    result["ram"]["after_execution"] = process.resourceUsage().maxRSS * 1024;

    if (error) {
        result["errors"]["js"] = error.message;
        result["exit_code"] = error.code;
    } else {
        result["exit_code"] = 0;
    }

    if (stderr) {
        result["stderr"] = stderr;
    }

    if (stdout) {
        result["stdout"] = stdout;
    }

    console.log(JSON.stringify(result));
});

