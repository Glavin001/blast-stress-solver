// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "scene_pack.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace blast_demo
{
namespace
{

class Json
{
public:
    enum class Kind
    {
        Null,
        Boolean,
        Number,
        String,
        Array,
        Object
    };

    Kind kind{Kind::Null};
    bool boolean{false};
    double number{0.0};
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    const Json& at(const std::string& key) const
    {
        const auto found = object.find(key);
        if (kind != Kind::Object || found == object.end())
        {
            throw std::runtime_error("scene pack is missing JSON field '" + key + "'");
        }
        return found->second;
    }

    const Json* find(const std::string& key) const
    {
        if (kind != Kind::Object)
        {
            return nullptr;
        }
        const auto found = object.find(key);
        return found == object.end() ? nullptr : &found->second;
    }
};

class JsonParser
{
public:
    explicit JsonParser(const std::string& input) : m_input(input) {}

    Json parse()
    {
        skipWhitespace();
        Json value = parseValue();
        skipWhitespace();
        if (m_position != m_input.size())
        {
            fail("trailing JSON data");
        }
        return value;
    }

private:
    const std::string& m_input;
    std::size_t m_position{0};

    [[noreturn]] void fail(const std::string& message) const
    {
        throw std::runtime_error(
            "invalid scene pack JSON at byte " + std::to_string(m_position) + ": " + message);
    }

    void skipWhitespace()
    {
        while (m_position < m_input.size()
            && std::isspace(static_cast<unsigned char>(m_input[m_position])) != 0)
        {
            ++m_position;
        }
    }

    bool consume(char expected)
    {
        skipWhitespace();
        if (m_position < m_input.size() && m_input[m_position] == expected)
        {
            ++m_position;
            return true;
        }
        return false;
    }

    void expect(char expected)
    {
        if (!consume(expected))
        {
            fail(std::string("expected '") + expected + "'");
        }
    }

    Json parseValue()
    {
        skipWhitespace();
        if (m_position >= m_input.size())
        {
            fail("unexpected end of input");
        }
        switch (m_input[m_position])
        {
        case '{':
            return parseObject();
        case '[':
            return parseArray();
        case '"':
        {
            Json value;
            value.kind = Json::Kind::String;
            value.string = parseString();
            return value;
        }
        case 't':
            return parseLiteral("true", Json::Kind::Boolean, true);
        case 'f':
            return parseLiteral("false", Json::Kind::Boolean, false);
        case 'n':
            return parseLiteral("null", Json::Kind::Null, false);
        default:
            return parseNumber();
        }
    }

    Json parseLiteral(const char* literal, Json::Kind kind, bool boolean)
    {
        const std::size_t length = std::char_traits<char>::length(literal);
        if (m_input.compare(m_position, length, literal) != 0)
        {
            fail("invalid literal");
        }
        m_position += length;
        Json value;
        value.kind = kind;
        value.boolean = boolean;
        return value;
    }

    Json parseObject()
    {
        Json value;
        value.kind = Json::Kind::Object;
        expect('{');
        if (consume('}'))
        {
            return value;
        }
        for (;;)
        {
            skipWhitespace();
            if (m_position >= m_input.size() || m_input[m_position] != '"')
            {
                fail("expected object key");
            }
            std::string key = parseString();
            expect(':');
            value.object.emplace(std::move(key), parseValue());
            if (consume('}'))
            {
                return value;
            }
            expect(',');
        }
    }

    Json parseArray()
    {
        Json value;
        value.kind = Json::Kind::Array;
        expect('[');
        if (consume(']'))
        {
            return value;
        }
        for (;;)
        {
            value.array.push_back(parseValue());
            if (consume(']'))
            {
                return value;
            }
            expect(',');
        }
    }

    static void appendUtf8(std::string& output, unsigned codepoint)
    {
        if (codepoint <= 0x7f)
        {
            output.push_back(static_cast<char>(codepoint));
        }
        else if (codepoint <= 0x7ff)
        {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
        else
        {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    std::string parseString()
    {
        expect('"');
        std::string output;
        while (m_position < m_input.size())
        {
            const char ch = m_input[m_position++];
            if (ch == '"')
            {
                return output;
            }
            if (ch != '\\')
            {
                output.push_back(ch);
                continue;
            }
            if (m_position >= m_input.size())
            {
                fail("truncated string escape");
            }
            const char escaped = m_input[m_position++];
            switch (escaped)
            {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            case 'u':
            {
                if (m_position + 4 > m_input.size())
                {
                    fail("truncated unicode escape");
                }
                unsigned codepoint = 0;
                for (int i = 0; i < 4; ++i)
                {
                    const char digit = m_input[m_position++];
                    codepoint <<= 4;
                    if (digit >= '0' && digit <= '9') codepoint += unsigned(digit - '0');
                    else if (digit >= 'a' && digit <= 'f') codepoint += unsigned(digit - 'a' + 10);
                    else if (digit >= 'A' && digit <= 'F') codepoint += unsigned(digit - 'A' + 10);
                    else fail("invalid unicode escape");
                }
                appendUtf8(output, codepoint);
                break;
            }
            default:
                fail("unsupported string escape");
            }
        }
        fail("unterminated string");
    }

    Json parseNumber()
    {
        const char* start = m_input.c_str() + m_position;
        char* end = nullptr;
        errno = 0;
        const double parsed = std::strtod(start, &end);
        if (end == start || errno == ERANGE)
        {
            fail("invalid number");
        }
        m_position += static_cast<std::size_t>(end - start);
        Json value;
        value.kind = Json::Kind::Number;
        value.number = parsed;
        return value;
    }
};

void requireKind(const Json& value, Json::Kind expected, const char* field)
{
    if (value.kind != expected)
    {
        throw std::runtime_error(std::string("scene pack field '") + field + "' has the wrong type");
    }
}

float number(const Json& value, const char* field)
{
    requireKind(value, Json::Kind::Number, field);
    if (!std::isfinite(value.number)
        || value.number < -std::numeric_limits<float>::max()
        || value.number > std::numeric_limits<float>::max())
    {
        throw std::runtime_error(std::string("scene pack field '") + field + "' is not a finite float");
    }
    return static_cast<float>(value.number);
}

std::uint32_t index(const Json& value, const char* field)
{
    const double raw = value.number;
    requireKind(value, Json::Kind::Number, field);
    if (raw < 0.0 || raw > static_cast<double>(std::numeric_limits<std::uint32_t>::max())
        || std::floor(raw) != raw)
    {
        throw std::runtime_error(std::string("scene pack field '") + field + "' is not an index");
    }
    return static_cast<std::uint32_t>(raw);
}

std::string string(const Json& value, const char* field)
{
    requireKind(value, Json::Kind::String, field);
    return value.string;
}

physx::PxVec3 vec3(const Json& value, const char* field)
{
    requireKind(value, Json::Kind::Object, field);
    return physx::PxVec3(
        number(value.at("x"), "x"),
        number(value.at("y"), "y"),
        number(value.at("z"), "z"));
}

float optionalNumber(const Json& object, const char* key, float fallback)
{
    const Json* value = object.find(key);
    return value ? number(*value, key) : fallback;
}

std::string readTextFile(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("could not open scene pack: " + path);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::vector<physx::PxVec3> reduceConvexPoints(
    const std::vector<physx::PxVec3>& input,
    std::size_t limit)
{
    std::vector<physx::PxVec3> unique;
    unique.reserve(input.size());
    for (const physx::PxVec3& point : input)
    {
        const bool duplicate = std::any_of(unique.begin(), unique.end(), [&](const physx::PxVec3& other) {
            return (point - other).magnitudeSquared() <= 1.0e-12f;
        });
        if (!duplicate)
        {
            unique.push_back(point);
        }
    }
    if (unique.size() <= limit)
    {
        return unique;
    }

    std::vector<bool> selected(unique.size(), false);
    std::vector<physx::PxVec3> result;
    result.reserve(limit);
    auto addIndex = [&](std::size_t indexValue) {
        if (!selected[indexValue])
        {
            selected[indexValue] = true;
            result.push_back(unique[indexValue]);
        }
    };
    for (int axis = 0; axis < 3; ++axis)
    {
        std::size_t minimum = 0;
        std::size_t maximum = 0;
        for (std::size_t i = 1; i < unique.size(); ++i)
        {
            if (unique[i][axis] < unique[minimum][axis]) minimum = i;
            if (unique[i][axis] > unique[maximum][axis]) maximum = i;
        }
        addIndex(minimum);
        addIndex(maximum);
    }

    while (result.size() < limit)
    {
        std::size_t best = unique.size();
        float bestDistance = -1.0f;
        for (std::size_t i = 0; i < unique.size(); ++i)
        {
            if (selected[i])
            {
                continue;
            }
            float nearest = std::numeric_limits<float>::max();
            for (const physx::PxVec3& chosen : result)
            {
                nearest = std::min(nearest, (unique[i] - chosen).magnitudeSquared());
            }
            if (nearest > bestDistance)
            {
                bestDistance = nearest;
                best = i;
            }
        }
        if (best == unique.size())
        {
            break;
        }
        addIndex(best);
    }
    return result;
}

// `library` holds the pack's shapeLibrary entries, empty when the pack has
// none. A `kind: "shape"` collider is a reference into it: the fracturer stored
// each distinct shard once and named it, so a consumer can share geometry
// without comparing point arrays to work out which shards are alike.
SceneCollider parseCollider(
    const Json& json,
    const physx::PxVec3& size,
    const std::vector<const Json*>& library)
{
    requireKind(json, Json::Kind::Object, "nodeColliders[]");
    SceneCollider result;
    const std::string kind = string(json.at("kind"), "kind");
    if (kind == "cuboid")
    {
        result.kind = SceneColliderKind::Cuboid;
        result.halfExtents = vec3(json.at("halfExtents"), "halfExtents");
        return result;
    }
    if (kind == "shape")
    {
        const std::size_t which = index(json.at("shape"), "shape");
        if (which >= library.size())
        {
            throw std::runtime_error(
                "scene collider references shape " + std::to_string(which)
                + " but the library has only " + std::to_string(library.size())
                + " entries");
        }
        // Resolve against the library entry, which is itself an ordinary hull.
        // Recursing keeps one implementation of the point validation rather
        // than a second copy that could drift from it.
        return parseCollider(*library[which], size, {});
    }
    if (kind != "convex_hull")
    {
        throw std::runtime_error("unsupported scene collider kind: " + kind);
    }

    result.kind = SceneColliderKind::ConvexHull;
    const Json& values = json.at("points");
    requireKind(values, Json::Kind::Array, "points");
    if (values.array.size() < 12 || values.array.size() % 3 != 0)
    {
        throw std::runtime_error("convex hull points must contain at least four xyz triples");
    }
    physx::PxVec3 minimum(std::numeric_limits<float>::max());
    physx::PxVec3 maximum(-std::numeric_limits<float>::max());
    std::vector<physx::PxVec3> sourcePoints;
    sourcePoints.reserve(values.array.size() / 3);
    for (std::size_t i = 0; i < values.array.size(); i += 3)
    {
        const physx::PxVec3 point(
            number(values.array[i], "points[]"),
            number(values.array[i + 1], "points[]"),
            number(values.array[i + 2], "points[]"));
        minimum = minimum.minimum(point);
        maximum = maximum.maximum(point);
        sourcePoints.push_back(point);
    }
    result.sourcePointCount = static_cast<std::uint32_t>(sourcePoints.size());
    result.points = reduceConvexPoints(sourcePoints, 64);
    if (result.points.size() < 4)
    {
        throw std::runtime_error("convex hull contains fewer than four unique points");
    }
    result.halfExtents = (maximum - minimum) * 0.5f;
    if (result.halfExtents.x <= 0.0f || result.halfExtents.y <= 0.0f || result.halfExtents.z <= 0.0f)
    {
        result.halfExtents = size * 0.5f;
    }
    return result;
}

} // namespace

ScenePack loadScenePack(const std::string& path)
{
    const Json root = JsonParser(readTextFile(path)).parse();
    requireKind(root, Json::Kind::Object, "root");
    const std::uint32_t version = index(root.at("version"), "version");
    if (version != 1 && version != 2 && version != 3)
    {
        throw std::runtime_error(
            "unsupported ScenePack version " + std::to_string(version)
            + " (see SCENE_PACK_FORMAT.md)");
    }

    ScenePack pack;
    pack.version = version;
    pack.title = string(root.at("title"), "title");

    const Json& defaults = root.at("defaults");
    const Json& projectile = defaults.at("projectile");
    pack.projectileRadius = number(projectile.at("radius"), "radius");
    pack.projectileMass = number(projectile.at("mass"), "mass");
    pack.projectileSpeed = number(projectile.at("speed"), "speed");
    pack.projectileTtlSeconds = optionalNumber(projectile, "ttlMs", 8000.0f) / 1000.0f;

    const Json& solver = defaults.at("solver");
    pack.gravity = number(solver.at("gravity"), "gravity");

    // Parse one material entry. Negative tension/shear limits are left as-is;
    // the solver resolves the inheritance when the table is installed.
    const auto parseLimits = [](const Json& source) {
        StressLimits limits;
        limits.compressionElastic =
            number(source.at("compressionElastic"), "compressionElastic");
        limits.compressionFatal = number(source.at("compressionFatal"), "compressionFatal");
        limits.tensionElastic = number(source.at("tensionElastic"), "tensionElastic");
        limits.tensionFatal = number(source.at("tensionFatal"), "tensionFatal");
        limits.shearElastic = number(source.at("shearElastic"), "shearElastic");
        limits.shearFatal = number(source.at("shearFatal"), "shearFatal");
        return limits;
    };

    // v3 adds chunk crushing: an optional `crush` block per material and an
    // optional `m` per node. Both are v3-only: seeing them in a v2 pack means
    // the author expects behaviour the declared version does not have, which
    // is worth an error rather than a silent no-op.
    const bool allowCrush = version >= 3;

    if (version >= 2)
    {
        // v2: materials are mandatory. A structure must state what it is made
        // of rather than inheriting a placeholder — that silent inheritance is
        // exactly how a pack ends up an order of magnitude off its own claimed
        // material (SCENE_PACK_FORMAT.md, "v1 trap").
        const Json* materials = solver.find("materials");
        if (!materials || materials->kind != Json::Kind::Array
            || materials->array.empty())
        {
            throw std::runtime_error(
                "ScenePack v2 requires a non-empty defaults.solver.materials array");
        }
        pack.materials.reserve(materials->array.size());
        for (std::size_t i = 0; i < materials->array.size(); ++i)
        {
            const Json& source = materials->array[i];
            requireKind(source, Json::Kind::Object, "materials[]");
            SceneMaterial material;
            if (const Json* name = source.find("name"))
            {
                material.name =
                    name->kind == Json::Kind::String ? name->string : std::string();
            }
            if (material.name.empty())
            {
                material.name = "material" + std::to_string(i);
            }
            material.limits = parseLimits(source);
            if (!(material.limits.compressionElastic >= 0.0f)
                || !(material.limits.compressionFatal >= material.limits.compressionElastic))
            {
                throw std::runtime_error(
                    "material '" + material.name
                    + "' needs compressionFatal >= compressionElastic >= 0");
            }
            if (const Json* crush = source.find("crush"))
            {
                if (!allowCrush)
                {
                    throw std::runtime_error(
                        "material '" + material.name
                        + "' has a crush block, which requires ScenePack version 3");
                }
                requireKind(*crush, Json::Kind::Object, "crush");
                material.crush.enabled = true;
                material.crush.capPressure = number(crush->at("capPressure"), "capPressure");
                material.crush.cohesion = optionalNumber(*crush, "cohesion", 0.0f);
                material.crush.frictionSlope = optionalNumber(*crush, "frictionSlope", 0.0f);
                material.crush.crushEnergy = number(crush->at("crushEnergy"), "crushEnergy");
                material.crush.crushViscosity =
                    number(crush->at("crushViscosity"), "crushViscosity");
                material.crush.strainRateExponent =
                    optionalNumber(*crush, "strainRateExponent", 0.0f);
                material.crush.referenceStrainRate =
                    optionalNumber(*crush, "referenceStrainRate", 1.0f);
                material.crush.debrisMassFraction =
                    optionalNumber(*crush, "debrisMassFraction", 0.0f);
                material.crush.debrisFragmentCount = static_cast<std::uint32_t>(
                    optionalNumber(*crush, "debrisFragmentCount", 0.0f));

                if (!(material.crush.capPressure > 0.0f))
                {
                    throw std::runtime_error(
                        "material '" + material.name
                        + "' crush.capPressure must be > 0 (omit the crush block to disable)");
                }
                if (!(material.crush.crushEnergy > 0.0f))
                {
                    throw std::runtime_error(
                        "material '" + material.name + "' crush.crushEnergy must be > 0");
                }
                if (!(material.crush.crushViscosity > 0.0f))
                {
                    throw std::runtime_error(
                        "material '" + material.name + "' crush.crushViscosity must be > 0");
                }
                if (material.crush.cohesion < 0.0f || material.crush.frictionSlope < 0.0f)
                {
                    throw std::runtime_error(
                        "material '" + material.name
                        + "' crush.cohesion and crush.frictionSlope must be >= 0");
                }
                if (material.crush.debrisMassFraction < 0.0f
                    || material.crush.debrisMassFraction > 1.0f)
                {
                    throw std::runtime_error(
                        "material '" + material.name
                        + "' crush.debrisMassFraction must be in [0, 1]");
                }
            }
            pack.materials.push_back(std::move(material));
        }
        pack.stressLimits = pack.materials.front().limits;
        pack.stressLimitsAuthored = true;
    }
    else if (const Json* limits = solver.find("limits"))
    {
        // v1 with explicit limits: synthesize the one-entry table so every
        // downstream consumer sees the same shape regardless of pack version.
        pack.stressLimits = parseLimits(*limits);
        pack.materials.push_back(SceneMaterial{"pack-limits", pack.stressLimits});
        pack.stressLimitsAuthored = true;
    }
    else
    {
        // A pack without solver.limits gets StressLimits{} — 1 MPa elastic /
        // 2 MPa fatal. That is roughly a twelfth of concrete's compressive
        // strength, so the structure is far weaker than the material it is
        // presumably meant to represent, and the difference has to be absorbed
        // somewhere else (a smaller projectile, a lower contact scale) for the
        // scene to behave. Say so rather than letting an unstated material
        // masquerade as an authored one.
        std::fprintf(
            stderr,
            "[scene_pack] '%s' omits defaults.solver.limits; falling back to "
            "%.3g/%.3g Pa compression and %.3g/%.3g Pa tension elastic/fatal. "
            "These are placeholder values, not an authored material — add an "
            "explicit limits block so the pack states what it is made of.\n",
            path.c_str(),
            static_cast<double>(pack.stressLimits.compressionElastic),
            static_cast<double>(pack.stressLimits.compressionFatal),
            static_cast<double>(pack.stressLimits.tensionElastic),
            static_cast<double>(pack.stressLimits.tensionFatal));
        pack.materials.push_back(SceneMaterial{"unstated", pack.stressLimits});
    }

    const Json& physics = defaults.at("physics");
    pack.friction = number(physics.at("friction"), "friction");
    pack.restitution = number(physics.at("restitution"), "restitution");
    pack.contactForceScale = number(physics.at("contactForceScale"), "contactForceScale");

    const Json& scenario = root.at("scenario");
    const Json& nodes = scenario.at("nodes");
    const Json& bonds = scenario.at("bonds");
    const Json& sizes = scenario.at("nodeSizes");
    const Json& colliders = scenario.at("nodeColliders");
    // Optional: only packs exported with a bounded fracture-pattern count carry
    // one. Gathered up front so a dangling reference fails while loading rather
    // than as a node silently colliding with some other node's shape.
    std::vector<const Json*> shapeLibrary;
    if (const Json* library = scenario.find("shapeLibrary"))
    {
        requireKind(*library, Json::Kind::Array, "shapeLibrary");
        shapeLibrary.reserve(library->array.size());
        for (const Json& entry : library->array)
        {
            shapeLibrary.push_back(&entry);
        }
    }
    requireKind(nodes, Json::Kind::Array, "nodes");
    requireKind(bonds, Json::Kind::Array, "bonds");
    requireKind(sizes, Json::Kind::Array, "nodeSizes");
    requireKind(colliders, Json::Kind::Array, "nodeColliders");
    if (nodes.array.size() != sizes.array.size() || nodes.array.size() != colliders.array.size())
    {
        throw std::runtime_error("scene pack node, size, and collider counts do not match");
    }

    pack.nodes.reserve(nodes.array.size());
    for (std::size_t i = 0; i < nodes.array.size(); ++i)
    {
        const Json& source = nodes.array[i];
        SceneNode node;
        node.centroid = vec3(source.at("centroid"), "centroid");
        node.mass = number(source.at("mass"), "mass");
        node.volume = number(source.at("volume"), "volume");
        // Node material index; absent means 0. Same no-clamp rule as bonds:
        // a silent clamp turns a typo into a mysteriously indestructible chunk.
        if (const Json* material = source.find("m"))
        {
            if (!allowCrush)
            {
                throw std::runtime_error(
                    "node " + std::to_string(i)
                    + " has a material index, which requires ScenePack version 3");
            }
            node.material = index(*material, "m");
            if (node.material >= pack.materials.size())
            {
                throw std::runtime_error(
                    "scene pack node " + std::to_string(i)
                    + " references material " + std::to_string(node.material)
                    + " but the table has only " + std::to_string(pack.materials.size())
                    + " entries");
            }
        }
        const physx::PxVec3 size = vec3(sizes.array[i], "nodeSizes[]");
        node.collider = parseCollider(colliders.array[i], size, shapeLibrary);
        node.visualHalfExtents =
            node.collider.kind == SceneColliderKind::Cuboid ? node.collider.halfExtents : size * 0.5f;
        pack.nodes.push_back(std::move(node));
    }

    // Optional real render geometry, one entry per node (root-level field,
    // e.g. a Voronoi-fractured shard's actual hull). Absent, `[]`, or a
    // length mismatch leaves every node at `present=false` (render as its box
    // collider) rather than erroring — a pack that doesn't care about
    // non-box visuals shouldn't need to know this field exists.
    pack.nodeMeshes.assign(pack.nodes.size(), SceneMesh{});
    if (const Json* meshes = root.find("nodeMeshes"))
    {
        if (meshes->kind == Json::Kind::Array && meshes->array.size() == pack.nodes.size())
        {
            const auto flatVec3Array = [](const Json& values, const char* field) {
                requireKind(values, Json::Kind::Array, field);
                if (values.array.size() % 3 != 0)
                {
                    throw std::runtime_error(
                        std::string(field) + " must contain whole xyz triples");
                }
                std::vector<physx::PxVec3> result;
                result.reserve(values.array.size() / 3);
                for (std::size_t i = 0; i < values.array.size(); i += 3)
                {
                    result.emplace_back(
                        number(values.array[i], field),
                        number(values.array[i + 1], field),
                        number(values.array[i + 2], field));
                }
                return result;
            };
            for (std::size_t i = 0; i < meshes->array.size(); ++i)
            {
                const Json& entry = meshes->array[i];
                if (entry.kind != Json::Kind::Object)
                {
                    continue; // null / absent: this node has no mesh, use its box.
                }
                SceneMesh mesh;
                mesh.positions = flatVec3Array(entry.at("positions"), "nodeMeshes[].positions");
                mesh.normals = flatVec3Array(entry.at("normals"), "nodeMeshes[].normals");
                if (mesh.positions.size() != mesh.normals.size())
                {
                    throw std::runtime_error(
                        "nodeMeshes[" + std::to_string(i)
                        + "]: positions/normals vertex count mismatch");
                }
                const Json& indices = entry.at("indices");
                requireKind(indices, Json::Kind::Array, "nodeMeshes[].indices");
                if (indices.array.size() % 3 != 0)
                {
                    throw std::runtime_error(
                        "nodeMeshes[" + std::to_string(i) + "].indices must be whole triangles");
                }
                mesh.indices.reserve(indices.array.size());
                for (const Json& value : indices.array)
                {
                    mesh.indices.push_back(index(value, "nodeMeshes[].indices"));
                }
                mesh.present = !mesh.positions.empty() && !mesh.indices.empty();
                pack.nodeMeshes[i] = std::move(mesh);
            }
        }
    }

    // Optional: authored node roles, used only for reporting joint classes.
    const Json* nodeTypes = scenario.find("nodeTypes");
    if (nodeTypes != nullptr
        && nodeTypes->kind == Json::Kind::Array
        && nodeTypes->array.size() == pack.nodes.size())
    {
        pack.nodeTypes.reserve(pack.nodes.size());
        for (const Json& entry : nodeTypes->array)
        {
            pack.nodeTypes.push_back(
                entry.kind == Json::Kind::String ? entry.string : std::string());
        }
    }

    pack.bonds.reserve(bonds.array.size());
    for (const Json& source : bonds.array)
    {
        SceneBond bond;
        bond.node0 = index(source.at("node0"), "node0");
        bond.node1 = index(source.at("node1"), "node1");
        bond.centroid = vec3(source.at("centroid"), "centroid");
        bond.normal = vec3(source.at("normal"), "normal");
        bond.area = number(source.at("area"), "area");
        // Material index; absent means 0. Out of range is an authoring error,
        // not something to clamp — a silent clamp turns a typo into a
        // mysteriously strong joint.
        if (const Json* material = source.find("m"))
        {
            bond.material = index(*material, "m");
            if (bond.material >= pack.materials.size())
            {
                throw std::runtime_error(
                    "scene pack bond " + std::to_string(pack.bonds.size())
                    + " references material " + std::to_string(bond.material)
                    + " but the table has only " + std::to_string(pack.materials.size())
                    + " entries");
            }
        }
        if (bond.node0 >= pack.nodes.size() || bond.node1 >= pack.nodes.size())
        {
            throw std::runtime_error("scene pack bond references an invalid node");
        }
        pack.bonds.push_back(bond);
    }
    if (pack.nodes.empty() || pack.bonds.empty())
    {
        throw std::runtime_error("scene pack must contain nodes and bonds");
    }
    return pack;
}

} // namespace blast_demo
