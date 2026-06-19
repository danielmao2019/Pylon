// Eager-imports every display modality's frontend apis module (Vite
// import.meta.glob) so each modality's module-load self-registration runs before
// any layered render; new modalities are auto-discovered with no edit here.
import.meta.glob("data/viewer/utils/displays/**/ts/frontend/apis.ts", { eager: true });
